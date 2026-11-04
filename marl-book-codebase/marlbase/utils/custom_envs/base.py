import inspect
import gymnasium as gym
import numpy as np
from lbforaging.foraging.environment import Action, ForagingEnv, Player


class ForagingDecompReward(ForagingEnv):
    """
    Base level-based foraging environment with decomposed (vectorized) rewards.
    Safely overrides internal lbforaging spawning to prevent choice(0) and array truth errors.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        players=2,
        min_player_level=1,
        max_player_level=1,
        field_size=(6, 6),
        min_food_level=1,
        max_food_level=2,
        max_num_food=3,
        sight=6,
        max_episode_steps=50,
        force_coop=False,
        normalize_reward=False,
        num_semantic_channels=4,
        step_penalty=0.01,
        miscoordination_penalty=1.0,
        leftover_food=None,
        food_to_collect=None,
        vector_reward=False,
        channel_names=None,
        step_penalty_channel=None,
        **kwargs,
    ):
        self.num_reward_channels = num_semantic_channels
        self.step_penalty = float(step_penalty)
        self.miscoordination_penalty = float(miscoordination_penalty)
        self.food_items = []
        self.food_specs = []
        self.leftover_food = leftover_food
        self.food_to_collect = food_to_collect
        self.vector_reward = vector_reward
        self.channel_names = channel_names

        self.min_player_level = min_player_level
        self.max_player_level = max_player_level
        self.min_food_level = min_food_level
        self.max_food_level = max_food_level

        self.step_penalty_channel = (
            step_penalty_channel
            if step_penalty_channel is not None
            else (self.num_reward_channels - 1 if self.num_reward_channels > 0 else None)
        )

        env_kwargs = {
            "players": players,
            "min_player_level": min_player_level,
            "max_player_level": max_player_level,
            "field_size": field_size,
            "min_food_level": min_food_level,
            "max_food_level": max_food_level,
            "sight": sight,
            "max_episode_steps": max_episode_steps,
            "force_coop": force_coop,
            "normalize_reward": normalize_reward,
            "max_num_food": max_num_food,
            "max_food": max_num_food,
            "food_to_collect": food_to_collect,
            "vector_reward": vector_reward,
        }
        env_kwargs.update(kwargs)

        sig = inspect.signature(ForagingEnv.__init__)
        valid_kwargs = {k: v for k, v in env_kwargs.items() if k in sig.parameters}
        super().__init__(**valid_kwargs)

    # ----------------------------------------------------------------------
    # Override lbforaging internal spawning with safe routines
    # ----------------------------------------------------------------------

    def spawn_food(self, *args, **kwargs):
        """No-op: Custom food placement is handled by _setup_food_items."""
        pass

    def spawn_players(self, *args, **kwargs):
        """
        Safe player spawning that resolves per-agent levels whether passed as
        scalars, lists, or NumPy arrays (e.g., np.array([4, 4])).
        """
        raw_min = args[0] if len(args) >= 2 else getattr(self, "min_player_level", 1)
        raw_max = args[1] if len(args) >= 2 else (args[0] if len(args) == 1 else getattr(self, "max_player_level", 1))

        rng = getattr(self, "np_random", np.random)

        # 1. Assign player levels per-agent index
        for i, p in enumerate(self.players):
            p.reward = 0.0
            p.score = 0.0

            min_lvl = int(raw_min[i]) if hasattr(raw_min, "__getitem__") else int(raw_min)
            max_lvl = int(raw_max[i]) if hasattr(raw_max, "__getitem__") else int(raw_max)

            if min_lvl >= max_lvl:
                p.level = max_lvl
            else:
                level_pool = list(range(min_lvl, max_lvl + 1))
                p.level = int(rng.choice(level_pool))

        # 2. Place players randomly without collisions
        occupied = set()
        for p in self.players:
            valid_coords = [
                (r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if (r, c) not in occupied
            ]
            if not valid_coords:
                break
            idx = int(rng.integers(0, len(valid_coords))) if hasattr(rng, "integers") else int(rng.choice(len(valid_coords)))
            pos = valid_coords[idx]
            p.position = pos
            occupied.add(pos)

    # ----------------------------------------------------------------------
    # Grid Placement & Reset
    # ----------------------------------------------------------------------

    def _place_foods_with_min_dist(self, min_dist: int = 2, max_retries: int = 50):
        occupied = set(p.position for p in self.players)
        rng = getattr(self, "np_random", np.random)

        for d in range(min_dist, -1, -1):
            for _ in range(max_retries):
                available_coords = {
                    (r, c)
                    for r in range(self.rows)
                    for c in range(self.cols)
                    if (r, c) not in occupied
                }
                chosen_positions = []

                for _ in range(len(self.food_items)):
                    if not available_coords:
                        break

                    coords_list = list(available_coords)
                    idx = int(rng.integers(0, len(coords_list))) if hasattr(rng, "integers") else int(rng.choice(len(coords_list)))
                    pos = coords_list[idx]
                    chosen_positions.append(pos)

                    r, c = pos
                    coords_to_remove = {
                        (nr, nc)
                        for nr in range(r - d, r + d + 1)
                        for nc in range(c - d, c + d + 1)
                        if 0 <= nr < self.rows and 0 <= nc < self.cols
                    }
                    available_coords.difference_update(coords_to_remove)

                if len(chosen_positions) == len(self.food_items):
                    for item, pos in zip(self.food_items, chosen_positions):
                        item["pos"] = pos
                        self.field[pos] = item["level"]
                    return

        raise RuntimeError(
            f"[Env] Not enough space on {self.rows}x{self.cols} to place {len(self.food_items)} food items."
        )

    def _reward_shape(self, r):
        if self.vector_reward:
            return r
        else:
            return [float(p.reward) for p in self.players]

    def _populate_info_food_metrics(self, info):
        total_types = len(self.food_specs) if self.food_specs else len(self.food_items)
        foods_collected = [False] * total_types
        foods_spawned = [False] * total_types

        for f in self.food_items:
            ch_id = f.get("channel_id", 0)
            if ch_id < total_types:
                foods_spawned[ch_id] = True
                if not f["active"]:
                    foods_collected[ch_id] = True

        info["foods_collected"] = foods_collected
        info["foods_spawned"] = foods_spawned
        info["num_foods_collected"] = sum(not f["active"] for f in self.food_items)
        info["foods_left"] = sum(f["active"] for f in self.food_items)
        info["num_foods_spawned"] = len(self.food_items)

    def reset(self, **kwargs):
        res = super().reset(**kwargs)
        obs, info = res if (isinstance(res, tuple) and len(res) == 2) else (res, {})

        self._setup_food_items()

        for player in self.players:
            player.reward_vec = np.zeros(self.num_reward_channels, dtype=np.float32)

        self.episode_reward_vec = np.zeros(
            (len(self.players), self.num_reward_channels), dtype=np.float32
        )

        self._game_over = False
        info["reward_vec"] = [p.reward_vec.copy() for p in self.players]
        self._populate_info_food_metrics(info)

        self._gen_valid_moves()
        obs = self._make_gym_obs()
        return obs, info

    def _setup_food_items(self):
        self.food_items = []

    def _calculate_food_reward(self, food, loading_players):
        food_level = food["level"]
        total_loading_level = sum(p.level for p in loading_players)
        return {p: float(food_level * (p.level / total_loading_level)) for p in loading_players}

    def _calculate_failed_load_penalty(self, food, loading_players):
        return {}

    def step(self, actions):
        self.current_step += 1

        self._gen_valid_moves()
        actions = [Action(int(a)) for a in actions]
        actions = [
            a if a in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        curr_positions = [p.position for p in self.players]
        proposed_positions = []
        for p, a in zip(self.players, actions):
            r, c = p.position
            if a == Action.NORTH:
                proposed_positions.append((r - 1, c))
            elif a == Action.SOUTH:
                proposed_positions.append((r + 1, c))
            elif a == Action.WEST:
                proposed_positions.append((r, c - 1))
            elif a == Action.EAST:
                proposed_positions.append((r, c + 1))
            else:
                proposed_positions.append((r, c))

        new_positions = list(proposed_positions)
        collision_resolved = False
        while not collision_resolved:
            collision_resolved = True
            for i in range(len(self.players)):
                if new_positions[i] == curr_positions[i]:
                    continue

                nr, nc = new_positions[i]
                if not (0 <= nr < self.rows and 0 <= nc < self.cols) or (self.field[nr, nc] > 0):
                    new_positions[i] = curr_positions[i]
                    collision_resolved = False
                    continue

                for j in range(len(self.players)):
                    if i != j:
                        if new_positions[i] == new_positions[j]:
                            new_positions[i] = curr_positions[i]
                            new_positions[j] = curr_positions[j]
                            collision_resolved = False
                        elif (new_positions[i] == curr_positions[j] and new_positions[j] == curr_positions[i]):
                            new_positions[i] = curr_positions[i]
                            new_positions[j] = curr_positions[j]
                            collision_resolved = False
                        elif (new_positions[i] == curr_positions[j] and new_positions[j] == curr_positions[j]):
                            new_positions[i] = curr_positions[i]
                            collision_resolved = False

        for p, pos, a in zip(self.players, new_positions, actions):
            p.position = pos
            p.history.append(a)

        rewards = [np.zeros(self.num_reward_channels, dtype=np.float32) for _ in range(len(self.players))]

        if self.step_penalty > 0.0 and self.step_penalty_channel is not None:
            if 0 <= self.step_penalty_channel < self.num_reward_channels:
                for p_idx in range(len(self.players)):
                    rewards[p_idx][self.step_penalty_channel] = -abs(self.step_penalty)

        for food in self.food_items:
            fr, fc = food["pos"]
            if not food["active"] or self.field[fr, fc] == 0:
                continue

            adj_players = self.adjacent_players(fr, fc)
            loading_players = [p for p in adj_players if actions[self.players.index(p)] == Action.LOAD]
            total_loading_level = sum(p.level for p in loading_players)

            if loading_players and total_loading_level >= food["level"]:
                food["active"] = False
                self.field[fr, fc] = 0
                payout_dict = self._calculate_food_reward(food, loading_players)
                ch_idx = food["channel_id"]
                for p, val in payout_dict.items():
                    rewards[self.players.index(p)][ch_idx] += val
            elif loading_players:
                penalty_dict = self._calculate_failed_load_penalty(food, loading_players)
                ch_idx = food["channel_id"]
                for p, val in penalty_dict.items():
                    rewards[self.players.index(p)][ch_idx] += val

        for p_idx, p in enumerate(self.players):
            p.reward_vec = rewards[p_idx]
            p.reward = float(rewards[p_idx].sum())
            p.score += p.reward
            self.episode_reward_vec[p_idx] += rewards[p_idx]

        num_collected = sum(not f["active"] for f in self.food_items)
        num_left = sum(f["active"] for f in self.food_items)

        if self.leftover_food is not None:
            terminated = bool(num_left <= self.leftover_food or self.field.sum() == 0)
        elif self.food_to_collect is not None:
            terminated = bool(num_collected >= self.food_to_collect or self.field.sum() == 0)
        else:
            terminated = bool(num_left == 0 or self.field.sum() == 0)

        truncated = bool(self.current_step >= self._max_episode_steps and not terminated)
        self._game_over = bool(terminated or truncated)

        self._gen_valid_moves()
        obs = self._make_gym_obs()

        info = self._get_info() if hasattr(self, "_get_info") else {}
        info["reward_vec"] = [r.copy() for r in rewards]
        self._populate_info_food_metrics(info)

        if terminated or truncated:
            team_channels = self.episode_reward_vec.sum(axis=0)
            for ch_idx in range(self.num_reward_channels):
                name = (
                    self.channel_names[ch_idx]
                    if self.channel_names and ch_idx < len(self.channel_names)
                    else f"channel_{ch_idx}"
                )
                info[f"return_{name}"] = float(team_channels[ch_idx])

        return obs, self._reward_shape(rewards), terminated, truncated, info