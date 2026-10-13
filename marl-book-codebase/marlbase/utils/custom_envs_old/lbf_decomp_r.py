import gymnasium as gym
import numpy as np
from lbforaging.foraging.environment import Action, ForagingEnv, Player


class ForagingDecompReward(ForagingEnv):
    """Base level-based foraging environment with decomposed (vectorized) rewards."""

    def __init__(
        self,
        *args,
        num_semantic_channels=4,
        step_penalty=0.01,
        normalize_reward=False,
        **kwargs,
    ):
        # Force normalize_reward=False to preserve absolute semantic payouts
        super().__init__(*args, normalize_reward=False, **kwargs)
        self.num_reward_channels = num_semantic_channels
        self.step_penalty = float(step_penalty)
        self.food_items = []

    def reset(self, **kwargs):
        res = super().reset(**kwargs)
        obs, info = res if (isinstance(res, tuple) and len(res) == 2) else (res, {})

        # Subclass hook to spawn and tag foods correctly every episode
        self._setup_food_items()

        for player in self.players:
            player.reward_vec = np.zeros(
                self.num_reward_channels, dtype=np.float32
            )

        self._game_over = False
        info["reward_vec"] = [p.reward_vec.copy() for p in self.players]
        info["foods_collected"] = [not f["active"] for f in self.food_items]

        # Return updated observation containing semantic features
        obs = self._make_decomposed_obs()
        return obs, info

    def _setup_food_items(self):
        """Default spatial allocation (overridden by specific testbeds)."""
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

    def _make_decomposed_obs(self):
        """Augments standard observations with semantic food metadata."""
        return self._make_gym_obs()

    def _calculate_food_reward(self, food, loading_players):
        """Hook for subclasses to compute semantic values."""
        food_level = food["level"]
        total_loading_level = sum(p.level for p in loading_players)
        payouts = {}
        for p in loading_players:
            payouts[p] = float(food_level * (p.level / total_loading_level))
        return payouts

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

        # 3. Vector Reward Initialization & Step Cost Application
        rewards = [
            np.zeros(self.num_reward_channels, dtype=np.float32)
            for _ in range(len(self.players))
        ]

        # Apply Step Cost on Channel Index 3
        if self.step_penalty > 0.0 and self.num_reward_channels >= 4:
            for p_idx in range(len(self.players)):
                rewards[p_idx][3] = -abs(self.step_penalty)

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

            if loading_players and total_loading_level >= food["level"]:
                food["active"] = False
                self.field[fr, fc] = 0

                payout_dict = self._calculate_food_reward(food, loading_players)
                ch_idx = food["channel_id"]
                for p, val in payout_dict.items():
                    p_idx = self.players.index(p)
                    rewards[p_idx][ch_idx] += val

        # 5. State Updates & Termination
        for p_idx, p in enumerate(self.players):
            p.reward_vec = rewards[p_idx]
            p.reward = float(rewards[p_idx].sum())
            p.score += p.reward

        terminated = bool(self.field.sum() == 0)
        truncated = bool(
            self.current_step >= self._max_episode_steps and not terminated
        )
        self._game_over = bool(terminated or truncated)

        self._gen_valid_moves()
        obs = self._make_decomposed_obs()

        info = self._get_info()
        info["reward_vec"] = [r.copy() for r in rewards]
        info["foods_collected"] = [not f["active"] for f in self.food_items]

        return obs, rewards, terminated, truncated, info