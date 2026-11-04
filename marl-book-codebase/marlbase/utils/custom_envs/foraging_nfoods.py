import copy
import numpy as np
from utils.custom_envs.base import ForagingDecompReward


class ForagingNFoods(ForagingDecompReward):
    """
    Unified Level-Based Foraging environment supporting N semantically distinct foods.
    Food items can be configured directly or chosen via presets ('3foods' or '4foods').
    """

    PRESETS = {
        "3foods": {
            "food_items": [
                {
                    "channel_id": 0,
                    "pos": None,
                    "level": 3,
                    "type": "high_coop",
                    "reward": 3.0,
                    "has_penalty": True,
                    "active": True,
                },
                {
                    "channel_id": 1,
                    "pos": None,
                    "level": 1,
                    "type": "low_det",
                    "reward": 1.0,
                    "has_penalty": False,
                    "active": True,
                },
                {
                    "channel_id": 2,
                    "pos": None,
                    "level": 2,
                    "type": "low_stoch",
                    "stoch_outcomes": [-2.0, 1.0],
                    "stoch_probs": [0.5, 0.5],
                    "has_penalty": False,
                    "active": True,
                },
            ],
            "channel_names": ["coop_food", "det_food", "stoch_food", "step_penalty"],
        },
        "4foods": {
            "food_items": [
                {
                    "channel_id": 0,
                    "pos": None,
                    "level": 3,
                    "type": "coop_with_penalty",
                    "reward": 4.0,
                    "has_penalty": True,
                    "active": True,
                },
                {
                    "channel_id": 1,
                    "pos": None,
                    "level": 3,
                    "type": "coop_no_penalty",
                    "reward": 2.5,
                    "has_penalty": False,
                    "active": True,
                },
                {
                    "channel_id": 2,
                    "pos": None,
                    "level": 1,
                    "type": "low_stoch",
                    "stoch_outcomes": [-2.5, 2.0],
                    "stoch_probs": [0.5, 0.5],
                    "has_penalty": False,
                    "active": True,
                },
                {
                    "channel_id": 3,
                    "pos": None,
                    "level": 1,
                    "type": "low_det",
                    "reward": 1.2,
                    "has_penalty": False,
                    "active": True,
                },
            ],
            "channel_names": ["coop_pen", "coop_free", "stoch", "det", "step_penalty"],
        },
    }

    def __init__(
        self,
        food_preset="4foods",
        food_items=None,
        min_food_dist=2,
        players=2,
        min_player_level=2,
        max_player_level=2,
        field_size=(6, 6),
        sight=6,
        max_episode_steps=50,
        force_coop=False,
        step_penalty=0.01,
        miscoordination_penalty=0.0,
        food_to_collect=None,
        vector_reward=False,
        **kwargs,
    ):
        self.min_food_dist = min_food_dist

        # 1. Resolve Food Specs and Channel Names
        if food_items is not None:
            self.food_specs = food_items
            channel_names = kwargs.pop("channel_names", None)
            if channel_names is None:
                channel_names = [f["type"] for f in self.food_specs] + ["step_penalty"]
        elif food_preset in self.PRESETS:
            preset = self.PRESETS[food_preset]
            self.food_specs = preset["food_items"]
            channel_names = kwargs.pop("channel_names", preset["channel_names"])
        else:
            raise ValueError(f"Unknown food_preset '{food_preset}'. Available: {list(self.PRESETS.keys())}")

        num_foods = len(self.food_specs)
        min_f_level = min(f["level"] for f in self.food_specs)
        max_f_level = max(f["level"] for f in self.food_specs)

        # Default num_semantic_channels: 1 channel per food + 1 channel for step penalty
        num_semantic_channels = kwargs.pop("num_semantic_channels", num_foods + 1)
        max_num_food = kwargs.pop("max_num_food", num_foods)

        if food_to_collect is None:
            food_to_collect = num_foods - 1

        super().__init__(
            players=players,
            min_player_level=min_player_level,
            max_player_level=max_player_level,
            field_size=field_size,
            min_food_level=min_f_level,
            max_food_level=max_f_level,
            sight=sight,
            max_episode_steps=max_episode_steps,
            force_coop=force_coop,
            max_num_food=max_num_food,
            step_penalty=step_penalty,
            miscoordination_penalty=miscoordination_penalty,
            num_semantic_channels=num_semantic_channels,
            food_to_collect=food_to_collect,
            vector_reward=vector_reward,
            channel_names=channel_names,
            step_penalty_channel=num_foods,
            **kwargs,
        )

        assert len(self.players) == 2, "Testbed requires exactly 2 agents."

    def _setup_food_items(self):
        """Spawns food items defined in init and places them with distance constraints."""
        self.field[:] = 0
        self.food_items = [copy.deepcopy(item) for item in self.food_specs]
        for item in self.food_items:
            item["pos"] = None
            item["active"] = True

        self._place_foods_with_min_dist(min_dist=self.min_food_dist)

    def _calculate_food_reward(self, food, loading_players):
        food_type = food["type"]
        payouts = {}

        if food_type in ("high_coop", "coop_with_penalty", "coop_no_penalty"):
            val = food.get("reward", 3.0)
            for p in loading_players:
                payouts[p] = float(val)

        elif food_type == "low_det":
            total_val = food.get("reward", 1.0)
            split_val = float(total_val / len(loading_players))
            for p in loading_players:
                payouts[p] = split_val

        elif food_type == "low_stoch":
            rng = getattr(self, "np_random", np.random)
            outcomes = food.get("stoch_outcomes", [-2.0, 1.0])
            probs = food.get("stoch_probs", [0.5, 0.5])
            sampled_val = float(rng.choice(outcomes, p=probs))
            for p in loading_players:
                payouts[p] = sampled_val

        else:
            food_level = food["level"]
            total_loading_level = sum(p.level for p in loading_players)
            for p in loading_players:
                payouts[p] = float(food_level * (p.level / total_loading_level))

        return payouts

    def _calculate_failed_load_penalty(self, food, loading_players):
        has_penalty = food.get(
            "has_penalty", food["type"] in ("high_coop", "coop_with_penalty")
        )
        penalties = {}
        if has_penalty:
            for p in loading_players:
                penalties[p] = -abs(self.miscoordination_penalty)
        return penalties


# Optional aliases for backwards compatibility
class Foraging3Foods(ForagingNFoods):
    def __init__(self, **kwargs):
        kwargs.setdefault("food_preset", "3foods")
        super().__init__(**kwargs)


class Foraging4FoodsStandard(ForagingNFoods):
    def __init__(self, **kwargs):
        kwargs.setdefault("food_preset", "4foods")
        super().__init__(**kwargs)