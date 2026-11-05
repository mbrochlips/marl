import numpy as np
from marlbase.utils.custom_envs_new.base import ForagingDecompReward


class Foraging4FoodsStandard(ForagingDecompReward):
    """
    Standard LBF with 4 distinct foods:
      - Channel 0: Coop Food with penalty (Level 3)
      - Channel 1: Coop Food without penalty (Level 3)
      - Channel 2: Stochastic Solo Food (Level 1)
      - Channel 3: Deterministic Solo Food (Level 1)
      - Channel 4: Step Cost
    """
    REWARD_CHANNELS = ["coop_pen", "coop_free", "stochv2", "detv2", "step_cost"]

    def __init__(
        self,
        players=2,
        min_player_level=2,
        max_player_level=2,
        field_size=(6, 6),
        min_food_level=1,
        max_food_level=3,
        force_coop=False,
        max_num_food=4,
        sight=6,
        max_episode_steps=50,
        step_penalty=0.01,
        miscoordination_penalty=0.4,
        num_semantic_channels=5,
        food_to_collect=3,
        vector_reward=True,
        **kwargs,
    ):
        super().__init__(
            players=players,
            min_player_level=min_player_level,
            max_player_level=max_player_level,
            field_size=field_size,
            min_food_level=min_food_level,
            max_food_level=max_food_level,
            sight=sight,
            max_episode_steps=max_episode_steps,
            force_coop=force_coop,
            max_num_food=max_num_food,
            step_penalty=step_penalty,
            miscoordination_penalty=miscoordination_penalty,
            num_semantic_channels=num_semantic_channels,
            food_to_collect=food_to_collect,
            vector_reward=vector_reward,
            **kwargs,
        )
        assert len(self.players) == 2, "Agent count must remain 2."

    def _setup_food_items(self):
        self.field[:] = 0
        self.food_items = [
            {
                "channel_id": 0,
                "pos": None,
                "level": 3,
                "type": "coop_with_penalty",
                "active": True,
            },
            {
                "channel_id": 1,
                "pos": None,
                "level": 3,
                "type": "coop_no_penalty",
                "active": True,
            },
            {
                "channel_id": 2,
                "pos": None,
                "level": 1,
                "type": "low_stoch",
                "active": True,
            },
            {
                "channel_id": 3,
                "pos": None,
                "level": 1,
                "type": "low_det",
                "active": True,
            },
        ]
        # Distance-constrained placement (handles 6x6 edge boundaries and spacing)
        self._place_foods_with_min_dist(min_dist=2)

    def _calculate_food_reward(self, food, loading_players):
        food_type = food["type"]
        payouts = {}

        if food_type == "coop_with_penalty":
            for p in loading_players:
                payouts[p] = 4.0
        elif food_type == "coop_no_penalty":
            for p in loading_players:
                payouts[p] = 2.5
        elif food_type == "low_stoch":
            rng = getattr(self, "np_random", np.random)
            val = float(rng.choice([-2.5, 2.0], p=[0.5, 0.5]))
            for p in loading_players:
                payouts[p] = val
        elif food_type == "low_det":
            val = 1.2 / len(loading_players)
            for p in loading_players:
                payouts[p] = val

        return payouts

    def _calculate_failed_load_penalty(self, food, loading_players):
        food_type = food["type"]
        penalties = {}
        if food_type == "coop_with_penalty":
            for p in loading_players:
                penalties[p] = -abs(self.miscoordination_penalty)
        return penalties

    def step(self, actions):
        obs, rew, terminated, truncated, info = super().step(actions)

        # Handle 5th channel (step penalty)
        if self.step_penalty > 0.0 and self.num_reward_channels == 5:
            for p_idx in range(len(self.players)):
                if self.vector_reward:
                    rew[p_idx][4] = -abs(self.step_penalty)

        if terminated or truncated:
            team_channels = self.episode_reward_vec.sum(axis=0)
            names = ["coop_pen", "coop_free", "stoch", "det", "step_penalty"]
            for idx in range(self.num_reward_channels):
                name = names[idx] if idx < len(names) else f"ch_{idx}"
                info[f"return_{name}"] = float(team_channels[idx])

        return obs, rew, terminated, truncated, info