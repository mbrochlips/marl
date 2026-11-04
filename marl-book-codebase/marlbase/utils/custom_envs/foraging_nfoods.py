import copy
import numpy as np
from marlbase.utils.custom_envs.base import ForagingDecompReward
from marlbase.utils.custom_envs.presets import FOOD_PRESETS


class ForagingNFoods(ForagingDecompReward):
    """
    Unified Level-Based Foraging environment supporting N semantically distinct foods.
    Food presets are imported from utils.custom_envs.presets.
    """
    PRESETS = FOOD_PRESETS

    def __init__(
        self,
        food_preset="4foods",
        foods=None,
        food_items=None,
        min_spawn=None,
        max_spawn=None,
        leftover_food=None,
        min_food_dist=2,
        players=2,
        min_player_level=None,
        max_player_level=None,
        field_size=None,
        sight=None,
        max_episode_steps=50,
        force_coop=False,
        step_penalty=0.01,
        miscoordination_penalty=0.0,
        vector_reward=False,
        **kwargs,
    ):
        self.min_food_dist = min_food_dist

        raw_items = foods if foods is not None else food_items

        if raw_items is not None:
            raw_specs = raw_items
            preset_min_spawn = len(raw_specs)
            preset_max_spawn = len(raw_specs)
            preset_leftover = 1
            default_field_size = (6, 6)
            default_sight = 6
            default_min_player_level = 2
            default_max_player_level = 2
            preset_channel_names = [f["type"] for f in raw_specs] + ["step_penalty"]
        elif food_preset in self.PRESETS:
            preset = self.PRESETS[food_preset]
            raw_specs = preset.get("foods", preset.get("food_items", []))
            preset_min_spawn = preset.get("min_spawn", len(raw_specs))
            preset_max_spawn = preset.get("max_spawn", len(raw_specs))
            preset_leftover = preset.get("leftover_food", 1)
            default_field_size = preset.get("field_size", (6, 6))
            default_sight = preset.get("sight", 6)
            default_min_player_level = preset.get("min_player_level", 2)
            default_max_player_level = preset.get("max_player_level", 2)
            preset_channel_names = preset.get(
                "channel_names",
                [f["type"] for f in raw_specs] + ["step_penalty"]
            )
        else:
            raise ValueError(
                f"Unknown food_preset '{food_preset}'. Available: {list(self.PRESETS.keys())}"
            )

        assert len(raw_specs) > 0, f"Preset '{food_preset}' has no foods defined."

        self.food_specs = []
        for idx, item in enumerate(raw_specs):
            spec = copy.deepcopy(item)
            spec["channel_id"] = idx
            self.food_specs.append(spec)

        self.channel_names = kwargs.pop("channel_names", preset_channel_names)

        min_player_level = min_player_level if min_player_level is not None else default_min_player_level
        max_player_level = max_player_level if max_player_level is not None else default_max_player_level
        field_size = field_size if field_size is not None else default_field_size
        sight = sight if sight is not None else default_sight

        num_diff_items = len(self.food_specs)
        self.min_spawn = min_spawn if min_spawn is not None else preset_min_spawn
        self.max_spawn = max_spawn if max_spawn is not None else preset_max_spawn

        assert 1 <= self.min_spawn <= self.max_spawn <= num_diff_items, (
            f"Require 1 <= min_spawn ({self.min_spawn}) <= max_spawn ({self.max_spawn}) "
            f"<= total item types ({num_diff_items})"
        )

        self.leftover_food = leftover_food if leftover_food is not None else preset_leftover
        assert self.leftover_food < self.min_spawn, (
            f"leftover_food ({self.leftover_food}) must be strictly less than min_spawn ({self.min_spawn})"
        )

        min_f_level = min(f["level"] for f in self.food_specs)
        max_f_level = max(f["level"] for f in self.food_specs)

        max_num_food = kwargs.pop("max_num_food", self.max_spawn)
        num_semantic_channels = kwargs.pop("num_semantic_channels", num_diff_items + 1)
        step_penalty_channel = kwargs.pop("step_penalty_channel", num_diff_items)

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
            leftover_food=self.leftover_food,
            vector_reward=vector_reward,
            channel_names=self.channel_names,
            step_penalty_channel=step_penalty_channel,
            **kwargs,
        )

        assert len(self.players) == 2, "Testbed requires exactly 2 agents."

    def _setup_food_items(self):
        self.field[:] = 0
        rng = getattr(self, "np_random", np.random)
        num_diff = len(self.food_specs)

        if self.min_spawn == self.max_spawn == num_diff:
            chosen_specs = self.food_specs
        else:
            if hasattr(rng, "integers"):
                k = int(rng.integers(self.min_spawn, self.max_spawn + 1))
            else:
                k = int(rng.randint(self.min_spawn, self.max_spawn + 1))

            k = min(max(k, 1), num_diff)
            chosen_indices = rng.choice(num_diff, size=k, replace=False)
            chosen_specs = [self.food_specs[i] for i in chosen_indices]

        self.food_items = [
            {**copy.deepcopy(spec), "pos": None, "active": True}
            for spec in chosen_specs
        ]

        self._place_foods_with_min_dist(min_dist=self.min_food_dist)

    def _calculate_food_reward(self, food, loading_players):
        num_players = len(loading_players)

        if "stoch_outcomes" in food:
            rng = getattr(self, "np_random", np.random)
            outcomes = food["stoch_outcomes"]
            probs = food.get("stoch_probs", None)
            val = float(rng.choice(outcomes, p=probs))
        else:
            val = float(food.get("reward", food["level"]))

        if food.get("split", False):
            val /= num_players

        return {p: val for p in loading_players}

    def _calculate_failed_load_penalty(self, food, loading_players):
        if food.get("has_penalty", False):
            penalty = -abs(self.miscoordination_penalty)
            return {p: penalty for p in loading_players}
        return {}


class Foraging3Foods(ForagingNFoods):
    def __init__(self, **kwargs):
        kwargs.setdefault("food_preset", "3foods")
        super().__init__(**kwargs)


class Foraging4FoodsStandard(ForagingNFoods):
    def __init__(self, **kwargs):
        kwargs.setdefault("food_preset", "4foods")
        super().__init__(**kwargs)