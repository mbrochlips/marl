import inspect
import gymnasium as gym
import numpy as np
from lbforaging.foraging.environment import Action, ForagingEnv, Player


class ForagingDecompReward(ForagingEnv):
    """
    Base level-based foraging environment with decomposed (vectorized) rewards.
    Supports failed-loading / miscoordination penalties and arbitrary reward channels.
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
        self.food_to_collect = food_to_collect
        self.vector_reward = vector_reward
        self.channel_names = channel_names

        # Step penalty channel defaults to the last channel
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

    def _reward_shape(self, r):
        if self.vector_reward:
            return r
        else:
            return [float(p.reward) for p in self.players]

    def _place_foods_with_min_dist(self, min_dist: int = 2, max_retries: int = 50):
        """Places self.food_items ensuring min Chebyshev distance, with automatic fallback."""
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
                    pos = coords_list[rng.choice(len(coords_list))]
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
        info["foods_collected"] = [not f["active"] for f in self.food_items]

        self._gen_valid_moves()
        obs = self._make_gym_obs()
        return obs, info

    def _setup_food_items(self):
        """Default spatial allocation (overridden by subclasses)."""
        self.food_items = []
        food_id = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.field[r, c] > 0:
                    self.food_items.append({
                        "channel_id": food_id,
                        "pos": (r, c),
                        "level": int(self.field[r, c]),
                        "type": "default",
                        "active": True,
                    })
                    food_id += 1

    def _calculate_food_reward(self, food, loading_players):
        food_level = food["level"]
        total_loading_level = sum(p.level for p in loading_players)
        payouts = {}
        for p in loading_players:
            payouts[p] = float(food_level * (p.level / total_loading_level))
        return payouts

    def _calculate_failed_load_penalty(self, food, loading_players):
        return {}

    def step(self, actions):
        self.current_step += 1

        # 1. Action Validation
        self._gen_valid_moves()
        actions = [Action(a) for a in actions]
        actions = [
            a if a in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # 2. Movement & Collision Resolution
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
                if not (0 <= nr < self.rows and 0 <= nc < self.cols) or (
                    self.field[nr, nc] > 0
                ):
                    new_positions[i] = curr_positions[i]
                    collision_resolved = False
                    continue

                for j in range(len(self.players)):
                    if i != j:
                        if new_positions[i] == new_positions[j]:
                            new_positions[i] = curr_positions[i]
                            new_positions[j] = curr_positions[j]
                            collision_resolved = False
                        elif (
                            new_positions[i] == curr_positions[j]
                            and new_positions[j] == curr_positions[i]
                        ):
                            new_positions[i] = curr_positions[i]
                            new_positions[j] = curr_positions[j]
                            collision_resolved = False
                        elif (
                            new_positions[i] == curr_positions[j]
                            and new_positions[j] == curr_positions[j]
                        ):
                            new_positions[i] = curr_positions[i]
                            collision_resolved = False

        for p, pos, a in zip(self.players, new_positions, actions):
            p.position = pos
            p.history.append(a)

        # 3. Vector Reward Initialization & Step Cost
        rewards = [
            np.zeros(self.num_reward_channels, dtype=np.float32)
            for _ in range(len(self.players))
        ]

        if self.step_penalty > 0.0 and self.step_penalty_channel is not None:
            if 0 <= self.step_penalty_channel < self.num_reward_channels:
                for p_idx in range(len(self.players)):
                    rewards[p_idx][self.step_penalty_channel] = -abs(self.step_penalty)

        # 4. Loading Phase
        for food in self.food_items:
            fr, fc = food["pos"]
            if not food["active"] or self.field[fr, fc] == 0:
                continue

            adj_players = self.adjacent_players(fr, fc)
            loading_players = [
                p
                for p in adj_players
                if actions[self.players.index(p)] == Action.LOAD
            ]
            total_loading_level = sum(p.level for p in loading_players)

            # Scenario A: Successful Collection
            if loading_players and total_loading_level >= food["level"]:
                food["active"] = False
                self.field[fr, fc] = 0

                payout_dict = self._calculate_food_reward(food, loading_players)
                ch_idx = food["channel_id"]
                for p, val in payout_dict.items():
                    p_idx = self.players.index(p)
                    rewards[p_idx][ch_idx] += val

            # Scenario B: Failed Load Attempt (Miscoordination)
            elif loading_players:
                penalty_dict = self._calculate_failed_load_penalty(
                    food, loading_players
                )
                ch_idx = food["channel_id"]
                for p, val in penalty_dict.items():
                    p_idx = self.players.index(p)
                    rewards[p_idx][ch_idx] += val

        # 5. State Updates & Termination
        for p_idx, p in enumerate(self.players):
            p.reward_vec = rewards[p_idx]
            p.reward = float(rewards[p_idx].sum())
            p.score += p.reward
            self.episode_reward_vec[p_idx] += rewards[p_idx]

        num_collected = sum(not f["active"] for f in self.food_items)
        target = (
            self.food_to_collect
            if self.food_to_collect is not None
            else len(self.food_items)
        )
        terminated = bool(num_collected >= target or self.field.sum() == 0)
        truncated = bool(
            self.current_step >= self._max_episode_steps and not terminated
        )
        self._game_over = bool(terminated or truncated)

        self._gen_valid_moves()
        obs = self._make_gym_obs()

        info = self._get_info()
        info["reward_vec"] = [r.copy() for r in rewards]
        info["foods_collected"] = [not f["active"] for f in self.food_items]
        info["num_foods_collected"] = num_collected

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