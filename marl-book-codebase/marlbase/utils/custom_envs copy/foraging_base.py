import inspect
import gymnasium as gym
import numpy as np
from lbforaging.foraging.environment import Action, ForagingEnv

MOVES = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.WEST: (0, -1),
    Action.EAST: (0, 1),
}


class ForagingDecompReward(ForagingEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        players=2,
        min_player_level=1,
        max_player_level=1,
        field_size=(6, 6),
        sight=6,
        max_episode_steps=50,
        force_coop=False,
        normalize_reward=False,
        num_semantic_channels=4,
        step_penalty=0.01,
        collision_penalty=0.03,
        miscoordination_penalty=1.0,
        food_to_collect=None,
        vector_reward=False,
        channel_names=None,
        **kwargs,
    ):

        self.food_items = []
        self.food_to_collect = food_to_collect
        self.vector_reward = vector_reward
        self.channel_names = channel_names
        self.step_penalty = float(step_penalty) if step_penalty is not None else 0.0
        self.collision_penalty = float(collision_penalty) if collision_penalty is not None else 0.0
        self.miscoordination_penalty = float(miscoordination_penalty) if miscoordination_penalty is not None else 0.0
        self.channel_names = list(channel_names) if channel_names is not None else []
        self.num_reward_channels = len(self.channel_names) if self.channel_names else int(num_semantic_channels)
        self.step_penalty_channel = (
            self.channel_names.index("step_penalty")
            if "step_penalty" in self.channel_names
            else None
        )
        self.collision_penalty_channel = (
            self.channel_names.index("collision_penalty")
            if "collision_penalty" in self.channel_names
            else None
        )
        

        env_kwargs = {
            "players": players,
            "min_player_level": min_player_level,
            "max_player_level": max_player_level,
            "field_size": field_size,
            "sight": sight,
            "max_episode_steps": max_episode_steps,
            "force_coop": force_coop,
            "normalize_reward": normalize_reward,
            **kwargs,
        }
        sig = inspect.signature(ForagingEnv.__init__)
        super().__init__(**{k: v for k, v in env_kwargs.items() if k in sig.parameters})

    def _reward_shape(self, r):
        return r if self.vector_reward else [float(p.reward) for p in self.players]

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._setup_food_items()

        for player in self.players:
            player.reward_vec = np.zeros(self.num_reward_channels, dtype=np.float32)
        self.episode_reward_vec = np.zeros((len(self.players), self.num_reward_channels), dtype=np.float32)

        self._game_over = False
        self._gen_valid_moves()

        info = {
            "reward_vec": [p.reward_vec.copy() for p in self.players],
            "foods_collected": [False] * len(self.food_items),
            "num_foods_collected": 0,
        }
        return self._make_gym_obs(), info

    def _setup_food_items(self):
        """Overridden by child classes."""
        pass

    def _calculate_food_reward(self, food, loading_players):
        return {p: float(food["level"]) / len(loading_players) for p in loading_players}

    def _calculate_failed_load_penalty(self, food, loading_players):
        return {}

    def step(self, actions):
        self.current_step += 1
        self._gen_valid_moves()
        actions = [
            Action(int(a)) if Action(int(a)) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]
        rewards = [np.zeros(self.num_reward_channels, dtype=np.float32) for _ in self.players]

        # 1. Resolve Movements & Collisions
        curr_pos = [p.position for p in self.players]
        new_pos = [
            (curr_pos[i][0] + MOVES.get(a, (0, 0))[0], curr_pos[i][1] + MOVES.get(a, (0, 0))[1])
            for i, a in enumerate(actions)
        ]
        collided_players = set()
        resolved = False

        while not resolved:
            resolved = True

            # 1. Grid Boundaries and Food Obstacles
            for i, (r, c) in enumerate(new_pos):
                if new_pos[i] != curr_pos[i]:
                    if not (0 <= r < self.rows and 0 <= c < self.cols) or self.field[r, c] > 0:
                        new_pos[i] = curr_pos[i]
                        resolved = False

            # 2. Head-on Swaps (Agent i and j cross paths directly)
            for i in range(len(self.players)):
                for j in range(i + 1, len(self.players)):
                    if (
                        new_pos[i] != curr_pos[i]
                        and new_pos[j] != curr_pos[j]
                        and new_pos[i] == curr_pos[j]
                        and new_pos[j] == curr_pos[i]
                    ):
                        new_pos[i], new_pos[j] = curr_pos[i], curr_pos[j]
                        collided_players.update([i, j])
                        resolved = False

            # 3. Same-Tile Contests (>= 2 agents target the exact same cell)
            dest_claims = {}
            for i, pos in enumerate(new_pos):
                if pos != curr_pos[i]:
                    dest_claims.setdefault(pos, []).append(i)

            for pos, claimants in dest_claims.items():
                if len(claimants) > 1:
                    for i in claimants:
                        new_pos[i] = curr_pos[i]
                        collided_players.add(i)
                    resolved = False

            # 4. Walking into a Stationary Agent (Domino cascades)
            stationary_cells = {curr_pos[i] for i, pos in enumerate(new_pos) if pos == curr_pos[i]}
            for i, pos in enumerate(new_pos):
                if pos != curr_pos[i] and pos in stationary_cells:
                    new_pos[i] = curr_pos[i]
                    collided_players.add(i)  # Only penalize the walker
                    resolved = False

        for p, pos, a in zip(self.players, new_pos, actions):
            p.position, _ = pos, p.history.append(a)

        if self.collision_penalty > 0.0 and self.collision_penalty_channel is not None:
            for p_idx in collided_players:
                rewards[p_idx][self.collision_penalty_channel] -= abs(self.collision_penalty)

        # 2. Step Penalty
        if self.step_penalty > 0.0 and self.step_penalty_channel is not None:
            for r in rewards:
                r[self.step_penalty_channel] = -abs(self.step_penalty)

        # 3. Food Interactions
        for food in self.food_items:
            if not food["active"]:
                continue
            fr, fc = food["pos"]
            adj_players = self.adjacent_players(fr, fc)
            loaders = [p for p in adj_players if actions[self.players.index(p)] == Action.LOAD]
            if not loaders:
                continue

            ch = self.channel_names.index(food["type"])
            if sum(p.level for p in loaders) >= food["level"]:
                food["active"] = False
                self.field[fr, fc] = 0
                for p, val in self._calculate_food_reward(food, loaders).items():
                    rewards[self.players.index(p)][ch] += val
            else:
                for p, val in self._calculate_failed_load_penalty(food, loaders).items():
                    rewards[self.players.index(p)][ch] += val

        # 4. State & Termination
        for p_idx, p in enumerate(self.players):
            p.reward_vec = rewards[p_idx]
            p.reward = float(rewards[p_idx].sum())
            p.score += p.reward
            self.episode_reward_vec[p_idx] += rewards[p_idx]

        num_collected = sum(not f["active"] for f in self.food_items)
        target = min(self.food_to_collect, len(self.food_items)) if self.food_to_collect else len(self.food_items)
        terminated = bool(num_collected >= target or self.field.sum() == 0)
        truncated = bool(self.current_step >= self._max_episode_steps and not terminated)
        self._game_over = terminated or truncated

        self._gen_valid_moves()
        info = {
            "reward_vec": [r.copy() for r in rewards],
            "foods_collected": [not f["active"] for f in self.food_items],
            "num_foods_collected": num_collected,
        }
        if terminated or truncated:
            team_return = self.episode_reward_vec.sum(axis=0)
            for idx, val in enumerate(team_return):
                name = self.channel_names[idx] if self.channel_names else f"ch_{idx}"
                info[f"return_{name}"] = float(val)

        return self._make_gym_obs(), self._reward_shape(rewards), terminated, truncated, info