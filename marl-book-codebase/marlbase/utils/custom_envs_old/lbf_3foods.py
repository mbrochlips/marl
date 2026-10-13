import gymnasium as gym
import numpy as np
from lbforaging.foraging.environment import Action, ForagingEnv, Player
from utils.custom_envs.lbf_decomp_r import ForagingDecompReward


class Foraging3Foods(ForagingDecompReward):
    """
    Testbed with 3 semantically distinct foods:
      - Channel 0: High-Level Cooperative Food (Level 2, reward 3.0 each)
      - Channel 1: Low-Level Deterministic Solo Food (Level 1, reward 0.5)
      - Channel 2: Low-Level Stochastic Solo Food (Level 1, reward in {0.1, 0.9})
      - Channel 3: Step Penalty Channel (-c_step per step)
    """

    def __init__(self, *args, step_penalty=0.01, **kwargs):
        super().__init__(
            *args,
            num_semantic_channels=4,
            step_penalty=step_penalty,
            max_num_food=3,
            **kwargs,
        )
        assert len(self.players) == 2, "Testbed requires exactly 2 agents."

    def _setup_food_items(self):
        """
        Clears the random field from super().reset() and places exactly:
          - 1 Cooperative Food (Level 2)
          - 1 Deterministic Solo Food (Level 1)
          - 1 Stochastic Solo Food (Level 1)
        at non-overlapping, reproducible or randomized positions.
        """
        # Clear field to guarantee exact food levels
        self.field[:] = 0
        self.food_items = []

        # Find empty coordinates not occupied by players
        occupied = set(p.position for p in self.players)
        all_coords = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in occupied
        ]
        chosen_indices = np.random.choice(
            len(all_coords), size=3, replace=False
        )
        pos_coop = all_coords[chosen_indices[0]]
        pos_det = all_coords[chosen_indices[1]]
        pos_stoch = all_coords[chosen_indices[2]]

        # Place onto grid
        self.field[pos_coop] = 2
        self.field[pos_det] = 1
        self.field[pos_stoch] = 1

        # Register semantic food records
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
            # Using {0.1, 0.9} guarantees r > 0, preventing the "silent pickup" ambiguity
            sampled_val = float(np.random.choice([0.1, 0.9]))
            split_val = sampled_val / len(loading_players)
            for p in loading_players:
                payouts[p] = split_val

        return payouts