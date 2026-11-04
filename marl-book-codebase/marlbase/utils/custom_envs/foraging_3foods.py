import numpy as np
from marlbase.utils.custom_envs.base import ForagingDecompReward


class Foraging3Foods(ForagingDecompReward):
    """
    Testbed with 3 semantically distinct foods:
      - Channel 0: High-Level Cooperative Food (Level 3, reward 3.0 each; miscoordination penalty)
      - Channel 1: Low-Level Deterministic Solo Food (Level 1, reward 1.0 split)
      - Channel 2: Low-Level Stochastic Solo Food (Level 2, reward in {-2.0, 1.0})
      - Channel 3: Step Penalty Channel (-c_step per step)
    """
    REWARD_CHANNELS = ["coop_food", "det_food", "stoch_food", "step_cost"]

    def __init__(
        self,
        players=2,
        min_player_level=2,
        max_player_level=2,
        min_food_level=1,
        max_food_level=3,
        force_coop=False,
        max_num_food=3,
        step_penalty=0.01,
        miscoordination_penalty=0.0,
        num_semantic_channels=4,
        food_to_collect=2,
        vector_reward=False,
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
            miscoordination_penalty=miscoordination_penalty,
            num_semantic_channels=num_semantic_channels,
            food_to_collect=food_to_collect,
            vector_reward=vector_reward,
            **kwargs,
        )
        assert len(self.players) == 2, "Testbed requires exactly 2 agents."

    def _setup_food_items(self):
        self.field[:] = 0
        self.food_items = [
            {
                "channel_id": 0,
                "pos": None,
                "level": 3,
                "type": "high_coop",
                "active": True,
            },
            {
                "channel_id": 1,
                "pos": None,
                "level": 1,
                "type": "low_det",
                "active": True,
            },
            {
                "channel_id": 2,
                "pos": None,
                "level": 2,
                "type": "low_stoch",
                "active": True,
            },
        ]
        # Distance-constrained placement
        self._place_foods_with_min_dist(min_dist=2)

    def _calculate_food_reward(self, food, loading_players):
        food_type = food["type"]
        payouts = {}

        if food_type == "high_coop":
            for p in loading_players:
                payouts[p] = 3.0
        elif food_type == "low_det":
            split_val = 1.0 / len(loading_players)
            for p in loading_players:
                payouts[p] = split_val
        elif food_type == "low_stoch":
            rng = getattr(self, "np_random", np.random)
            sampled_val = float(rng.choice([-2.0, 1.0], p=[0.5, 0.5]))
            for p in loading_players:
                payouts[p] = sampled_val

        return payouts

    def _calculate_failed_load_penalty(self, food, loading_players):
        food_type = food["type"]
        penalties = {}
        if food_type == "high_coop":
            for p in loading_players:
                penalties[p] = -abs(self.miscoordination_penalty)
        return penalties