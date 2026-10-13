import inspect
import gymnasium as gym
import numpy as np
from lbforaging.foraging.environment import Action, ForagingEnv, Player


class ForagingDecompReward(ForagingEnv):
    """
    Base level-based foraging environment with decomposed (vectorized) rewards.
    Supplies all required LBF positional arguments by default.
    """

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
        **kwargs,
    ):
        self.num_reward_channels = num_semantic_channels
        self.step_penalty = float(step_penalty)
        self.food_items = []

        # Prepare all possible arguments required by various LBF versions
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
            "max_food": max_num_food,  # alias for older LBF forks
        }
        env_kwargs.update(kwargs)

        # Filter strictly by what ForagingEnv.__init__ actually accepts
        sig = inspect.signature(ForagingEnv.__init__)
        valid_kwargs = {
            k: v for k, v in env_kwargs.items() if k in sig.parameters
        }

        super().__init__(**valid_kwargs)

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

        # Re-generate obs so agent vision reflects the custom food placement
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

        # 3. Vector Reward Initialization & Step Cost
        rewards = [
            np.zeros(self.num_reward_channels, dtype=np.float32)
            for _ in range(len(self.players))
        ]

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
        obs = self._make_gym_obs()

        info = self._get_info()
        info["reward_vec"] = [r.copy() for r in rewards]
        info["foods_collected"] = [not f["active"] for f in self.food_items]

        return obs, rewards, terminated, truncated, info


class Foraging3Foods(ForagingDecompReward):
    """
    Testbed with 3 semantically distinct foods:
      - Channel 0: High-Level Cooperative Food (Level 2, reward 3.0 each)
      - Channel 1: Low-Level Deterministic Solo Food (Level 1, reward 0.5)
      - Channel 2: Low-Level Stochastic Solo Food (Level 1, reward in {0.1, 0.9})
      - Channel 3: Step Penalty Channel (-c_step per step)
    """

    def __init__(
        self,
        players=2,
        min_player_level=1,
        max_player_level=1,
        min_food_level=1,
        max_food_level=2,
        force_coop=False,
        max_num_food=3,
        step_penalty=0.01,
        num_semantic_channels=4,
        **kwargs,
    ):
        super().__init__(
            players=players,
            min_player_level=min_player_level,
            max_player_level=max_player_level,
            min_food_level=min_food_level,
            max_food_level=max_food_level,
            force_coop=force_coop,
            max_num_food=max_num_food,
            step_penalty=step_penalty,
            num_semantic_channels=num_semantic_channels,
            **kwargs,
        )
        assert len(self.players) == 2, "Testbed requires exactly 2 agents."

    def _setup_food_items(self):
        """
        Clears the randomly generated field and places:
          - 1 Cooperative Food (Level 2)
          - 1 Deterministic Solo Food (Level 1)
          - 1 Stochastic Solo Food (Level 1)
        at empty positions using the env's seeded np_random generator.
        """
        self.field[:] = 0
        self.food_items = []

        occupied = set(p.position for p in self.players)
        all_coords = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in occupied
        ]

        # Use self.np_random to respect the episode seed
        rng = getattr(self, "np_random", np.random)
        chosen_indices = rng.choice(len(all_coords), size=3, replace=False)
        pos_coop = all_coords[chosen_indices[0]]
        pos_det = all_coords[chosen_indices[1]]
        pos_stoch = all_coords[chosen_indices[2]]

        self.field[pos_coop] = 2
        self.field[pos_det] = 1
        self.field[pos_stoch] = 1

        self.food_items = [
            {
                "channel_id": 0,
                "pos": pos_coop,
                "level": 2,
                "type": "high_coop",
                "active": True,
            },
            {
                "channel_id": 1,
                "pos": pos_det,
                "level": 1,
                "type": "low_det",
                "active": True,
            },
            {
                "channel_id": 2,
                "pos": pos_stoch,
                "level": 1,
                "type": "low_stoch",
                "active": True,
            },
        ]

    def _calculate_food_reward(self, food, loading_players):
        food_type = food["type"]
        payouts = {}

        if food_type == "high_coop":
            for p in loading_players:
                payouts[p] = 3.0
        elif food_type == "low_det":
            split_val = 0.5 / len(loading_players)
            for p in loading_players:
                payouts[p] = split_val
        elif food_type == "low_stoch":
            rng = getattr(self, "np_random", np.random)
            sampled_val = float(rng.choice([0.1, 0.9]))
            split_val = sampled_val / len(loading_players)
            for p in loading_players:
                payouts[p] = split_val

        return payouts


# Register the environment with Gymnasium
gym.register(
    id="Foraging-3Foods-6x6-2p-3f-v0",
    entry_point="utils.custom_envs:Foraging3Foods",
    kwargs={
        "players": 2,
        "min_player_level": 1,
        "max_player_level": 1,
        "field_size": (6, 6),
        "min_food_level": 1,
        "max_food_level": 2,
        "max_num_food": 3,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "num_semantic_channels": 4,
    },
)