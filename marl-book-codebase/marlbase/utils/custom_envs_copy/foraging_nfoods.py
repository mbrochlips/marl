import copy
import numpy as np
from marlbase.utils.custom_envs.foraging_base import ForagingDecompReward
from marlbase.utils.custom_envs.food_presets import FOOD_PRESETS


class ForagingNFoods(ForagingDecompReward):
    PRESETS = FOOD_PRESETS

    def __init__(
        self,
        food_preset="3foods",
        food_items=None,
        min_food_dist=2,
        players=2,
        min_player_level=4,
        max_player_level=4,
        field_size=(6, 6),
        sight=8,
        max_episode_steps=50,
        force_coop=False,
        step_penalty=0.01,
        miscoordination_penalty=0.2,
        collision_penalty=0.0, #0.05
        food_to_collect=None,
        vector_reward=True,
        **kwargs,
    ):
        self.min_food_dist = min_food_dist

        # 1. Resolve Food Specs
        if food_items is not None:
            self.food_specs = food_items
        elif food_preset in self.PRESETS:
            self.food_specs = self.PRESETS[food_preset]["food_items"]
        else:
            raise ValueError(f"Unknown preset '{food_preset}'. Available: {list(self.PRESETS.keys())}")

        # 2. Build Single Source of Truth for Channel Names
        channel_names = kwargs.pop("channel_names", None)
        if channel_names is None:
            channel_names = [f["type"] for f in self.food_specs]
            if collision_penalty is not None and float(collision_penalty) > 0.0:
                channel_names.append("collision_penalty")
            if step_penalty is not None and float(step_penalty) > 0.0:
                channel_names.append("step_penalty")

        num_foods = len(self.food_specs)
        min_f_level = min(f["level"] for f in self.food_specs)
        max_f_level = max(f["level"] for f in self.food_specs)

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
            max_num_food=num_foods,
            step_penalty=step_penalty,
            miscoordination_penalty=miscoordination_penalty,
            collision_penalty=collision_penalty,
            num_semantic_channels=len(channel_names),
            food_to_collect=food_to_collect,
            vector_reward=vector_reward,
            channel_names=channel_names,
            **kwargs,
        )

    def _setup_food_items(self):
        """Randomly spawns food from presets until no more space (min_dist) or no more items."""
        self.field[:] = 0
        rng = getattr(self, "np_random", np.random)

        # 1. Deterministic initial coordinate grid avoiding agents
        occupied = {p.position for p in self.players}
        available_coords = [
            (r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in occupied
        ]

        # 2. Randomize candidate food order safely without mutating preset
        food_candidates = [copy.deepcopy(f) for f in self.food_specs]
        perm = rng.permutation(len(food_candidates))
        shuffled_candidates = [food_candidates[i] for i in perm]

        # 3. Greedy placement until no space or pool exhausted
        self.food_items = []
        for food in shuffled_candidates:
            if not available_coords:
                break

            idx = int(rng.choice(len(available_coords)))
            pos = available_coords[idx]

            food["pos"] = pos
            food["active"] = True
            self.field[pos] = food["level"]
            self.food_items.append(food)

            # Eliminate coordinates with Manhattan distance < min_food_dist
            r, c = pos
            available_coords = [
                (nr, nc) for (nr, nc) in available_coords
                if abs(nr - r) + abs(nc - c) > self.min_food_dist
            ]

    def _calculate_food_reward(self, food, loading_players):
        rng = getattr(self, "np_random", np.random)
        if "stoch_outcomes" in food:
            val = float(rng.choice(food["stoch_outcomes"], p=food.get("stoch_probs")))
        else:
            val = float(food.get("reward", food["level"]))

        payout = val / len(loading_players) if food.get("split", False) else val
        return {p: payout for p in loading_players}

    def _calculate_failed_load_penalty(self, food, loading_players):
        if food.get("has_penalty", False):
            return {p: -abs(self.miscoordination_penalty) for p in loading_players}
        return {}