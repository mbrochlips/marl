import copy
import numpy as np
from marlbase.utils.custom_envs.foraging_base import ForagingDecompReward
from marlbase.utils.custom_envs.food_presets import FOOD_PRESETS


class ForagingNFoods(ForagingDecompReward):
    PRESETS = FOOD_PRESETS

    def __init__(
        self,
        food_preset="8foods",
        food_items=None,
        min_food_dist=2,
        players=2,
        min_player_level=4,
        max_player_level=4,
        field_size=(8, 8),
        sight=8,
        max_episode_steps=50,
        force_coop=False,
        step_penalty=0.01,
        collision_penalty=0.0,
        miscoordination_penalty=0.2,
        food_to_collect=None,
        vector_reward=True,
        **kwargs,
    ):
        self.min_food_dist = min_food_dist

        if food_items is not None:
            self.food_specs = food_items
            channel_names = kwargs.pop("channel_names", [f["type"] for f in self.food_specs] + ["step_penalty"])
        elif food_preset in self.PRESETS:
            preset = self.PRESETS[food_preset]
            self.food_specs = preset["food_items"]
            channel_names = kwargs.pop("channel_names", preset["channel_names"])
        else:
            raise ValueError(f"Unknown preset '{food_preset}'. Available: {list(self.PRESETS.keys())}")

        total_preset_foods = len(self.food_specs)

        super().__init__(
            players=players,
            min_player_level=min_player_level,
            max_player_level=max_player_level,
            field_size=field_size,
            sight=sight,
            max_episode_steps=max_episode_steps,
            force_coop=force_coop,
            max_num_food=total_preset_foods,
            step_penalty=step_penalty,
            collision_penalty=collision_penalty,
            miscoordination_penalty=miscoordination_penalty,
            num_semantic_channels=total_preset_foods + 1,
            step_penalty_channel=total_preset_foods,
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

            # Eliminate coordinates with Chebyshev distance < min_food_dist
            r, c = pos
            d = self.min_food_dist
            available_coords = [
                (nr, nc) for (nr, nc) in available_coords
                # if max(abs(nr - r), abs(nc - c)) >= d
                if (abs(nr - r) + abs(nc - c)) > d
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