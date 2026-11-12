import copy
import numpy as np
from marlbase.utils.custom_envs.base_six import ForagingDecompReward
from marlbase.utils.custom_envs.food_presets_six import FOOD_PRESETS


class ForagingSixFoods(ForagingDecompReward):
    PRESETS = FOOD_PRESETS

    def __init__(
        self,
        food_preset="ALL",
        food_items=None,
        min_food_dist=2,
        num_spawn_foods=3,
        players=2,
        min_player_level=3,
        max_player_level=3,
        field_size=(6, 6),
        sight=8,
        max_episode_steps=50,
        force_coop=False,
        step_penalty=0.01,
        miscoordination_penalty=0.2,
        collision_penalty=0.0,
        food_to_collect=None,
        vector_reward=True,
        num_food_types: int = 6,
        normalize_coordinates: bool = True,
        observe_agent_levels: bool = False,
        flatten_obs: bool = False,
        shuffle_entities: bool = False,
        **kwargs,
    ):
        # 1. Resolve Food Items
        if food_items is not None:
            self.food_specs = food_items
        elif food_preset in self.PRESETS:
            self.food_specs = self.PRESETS[food_preset]["food_items"]
        else:
            raise ValueError(f"Unknown preset '{food_preset}'. Available: {list(self.PRESETS.keys())}")

        self.min_food_dist = min_food_dist
        self.num_spawn_foods = num_spawn_foods

        # 2. Canonical Channel Mapping
        channel_names = kwargs.pop("channel_names", None)
        if channel_names is None:
            channel_names = [f["type"] for f in self.food_specs]
            if collision_penalty is not None and float(collision_penalty) > 0.0:
                channel_names.append("collision_penalty")
            if step_penalty is not None and float(step_penalty) > 0.0:
                channel_names.append("step_penalty")

        max_food = self.num_spawn_foods if self.num_spawn_foods is not None else len(self.food_specs)

        super().__init__(
            players=players,
            min_player_level=min_player_level,
            max_player_level=max_player_level,
            field_size=field_size,
            sight=sight,
            max_episode_steps=max_episode_steps,
            force_coop=force_coop,
            max_food=max_food,
            step_penalty=step_penalty,
            miscoordination_penalty=miscoordination_penalty,
            collision_penalty=collision_penalty,
            food_to_collect=food_to_collect,
            vector_reward=vector_reward,
            channel_names=channel_names,
            num_food_types=num_food_types,
            normalize_coordinates=normalize_coordinates,
            observe_agent_levels=observe_agent_levels,
            flatten_obs=flatten_obs,
            shuffle_entities=shuffle_entities,
            **kwargs,
        )

    # def _setup_food_items(self):
    #     """Randomly spawns food with a retry loop to prevent distance-constraint starvation."""
    #     rng = getattr(self, "np_random", np.random)
    #     occupied = {p.position for p in self.players}

    #     max_attempts = 50
    #     for _ in range(max_attempts):
    #         self.field[:] = 0
    #         available_coords = [
    #             (r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in occupied
    #         ]

    #         food_candidates = [copy.deepcopy(f) for f in self.food_specs]
    #         perm = rng.permutation(len(food_candidates))
    #         shuffled_candidates = [food_candidates[i] for i in perm]

    #         placed_foods = []
    #         success = True

    #         for food in shuffled_candidates:
    #             if not available_coords:
    #                 success = False
    #                 break

    #             idx = int(rng.choice(len(available_coords)))
    #             pos = available_coords[idx]

    #             food["pos"] = pos
    #             food["active"] = True
    #             self.field[pos] = food["level"]
    #             placed_foods.append(food)

    #             r, c = pos
    #             d = self.min_food_dist
    #             available_coords = [
    #                 (nr, nc) for (nr, nc) in available_coords
    #                 if (abs(nr - r) + abs(nc - c)) > d
    #             ]

    #         if success and len(placed_foods) == len(self.food_specs):
    #             self.food_items = placed_foods
    #             return

    #     # Fallback if distance constraint was too aggressive: relax distance constraint
    #     self.food_items = placed_foods

    def _setup_food_items(self):
        """Randomly spawns food from presets without mutating specs."""
        self.field[:] = 0
        rng = getattr(self, "np_random", np.random)

        occupied = {p.position for p in self.players}
        available_coords = [
            (r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in occupied
        ]

        food_candidates = [copy.deepcopy(f) for f in self.food_specs]
        perm = rng.permutation(len(food_candidates))
        shuffled_candidates = [food_candidates[i] for i in perm]

        count = 0
        self.food_items = []
        for food in shuffled_candidates:
            if not available_coords:
                break
            if count >= self.num_spawn_foods:
                break

            idx = int(rng.choice(len(available_coords)))
            pos = available_coords[idx]

            food["pos"] = pos
            food["active"] = True
            self.field[pos] = food["level"]
            self.food_items.append(food)

            # Eliminate coordinates with Manhattan distance <= min_food_dist
            r, c = pos
            d = self.min_food_dist
            available_coords = [
                (nr, nc) for (nr, nc) in available_coords
                if (abs(nr - r) + abs(nc - c)) > d
            ]

            count += 1

        #if len(shuffled_candidates) != len(self.food_specs):
        #    print("[Env] Not all foods were spawned")

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