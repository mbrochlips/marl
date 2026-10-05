import numpy as np
import gymnasium as gym
from lbforaging.foraging.environment import Action
from lbf_decomp_r import ForagingDecompReward 


class Foraging3Foods(ForagingDecompReward):
    """
    LBF testbed with intuitive 1-to-1 Level-to-ID mapping:
      - Channel 0 ('ID_1'): Level 1 -> Stochastic Solo (2 >= 1),         r in {0.0, 1.0}
      - Channel 1 ('ID_2'): Level 2 -> Low-level Solo (2 >= 2),          r = 0.5
      - Channel 2 ('ID_3'): Level 3 -> High-level Coop (2 + 2 >= 3),     r = 1.0
    """

    REWARD_CHANNELS = ["ID_1", "ID_2", "ID_3"]

    def __init__(
        self,
        field_size=(8, 8),
        max_episode_steps=50,
        sight=8,
        penalty=0.0,
        render_mode=None,
        **kwargs,
    ):
        # Both agents are locked to Level 2
        super().__init__(
            players=2,
            min_player_level=2,
            max_player_level=2,
            min_food_level=1,
            max_food_level=3,
            field_size=field_size,
            max_num_food=3,
            sight=sight,
            max_episode_steps=max_episode_steps,
            force_coop=False,
            normalize_reward=False,
            penalty=penalty,
            render_mode=render_mode,
            **kwargs,
        )
        self.num_reward_channels = 3

    def spawn_food(self, *args, **kwargs):
        self.food_items = []
        player_positions = {p.position for p in self.players}
        available_coords = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in player_positions
        ]

        chosen_indices = self.np_random.choice(
            len(available_coords), size=3, replace=False
        )
        coords = [available_coords[i] for i in chosen_indices]

        # Natural 1-to-1 mapping: ID 1 -> Level 1, ID 2 -> Level 2, ID 3 -> Level 3
        food_specs = [
            {"id": 0, "name": "ID_1", "desc": "Stochastic Solo", "level": 1},
            {"id": 1, "name": "ID_2", "desc": "Low-level Solo",  "level": 2},
            {"id": 2, "name": "ID_3", "desc": "High-level Coop", "level": 3},
        ]

        for spec, (r, c) in zip(food_specs, coords):
            self.field[r, c] = spec["level"]
            self.food_items.append({
                "id": spec["id"],
                "name": spec["name"],
                "desc": spec["desc"],
                "pos": (r, c),
                "level": spec["level"],
                "active": True,
            })

    def _calculate_food_reward(self, food, loading_players):
        if food["id"] == 0:
            # ID_1 (Level 1): Stochastic Solo
            payout = float(self.np_random.choice([0.0, 1.0]))
        elif food["id"] == 1:
            # ID_2 (Level 2): Low-level Solo
            payout = 0.5
        elif food["id"] == 2:
            # ID_3 (Level 3): High-level Coop
            payout = 1.0
        else:
            payout = 0.0

        return {p: payout for p in loading_players}

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["food_positions"] = {f["name"]: f["pos"] for f in self.food_items}
        info["channel_names"] = self.REWARD_CHANNELS
        return obs, info


if __name__ == "__main__":
    env = Foraging3Foods(field_size=(6, 6), max_episode_steps=50)

    # -------------------------------------------------------------
    # TEST 1: Random Placements Across Resets
    # -------------------------------------------------------------
    print("--- TEST 1: Random Placements Across Resets ---")
    for ep in range(2):
        obs, info = env.reset()
        print(f"Episode {ep + 1}:")
        print(f"  Agent positions: {[p.position for p in env.players]}")
        for f in env.food_items:
            print(f"  Food {f['name']} ({f['desc']}) at {f['pos']} [Level {f['level']}]")

    # -------------------------------------------------------------
    # TEST 2: Deterministic Mechanics Test (Coop & Solo)
    # -------------------------------------------------------------
    print("\n--- TEST 2: Deterministic Mechanics Test ---")
    env.reset()
    env.field.fill(0)

    # Food 0 (ID_1, Stochastic Solo, Level 1) far away at (5, 5)
    env.food_items[0]["pos"] = (5, 5)
    env.food_items[0]["active"] = True
    env.field[5, 5] = 1

    # Food 1 (ID_2, Low-level Solo, Level 2) far away at (0, 0)
    env.food_items[1]["pos"] = (0, 0)
    env.food_items[1]["active"] = True
    env.field[0, 0] = 2

    # Food 2 (ID_3, High-level Coop, Level 3) placed at (2, 2)
    env.food_items[2]["pos"] = (2, 2)
    env.food_items[2]["active"] = True
    env.field[2, 2] = 3

    # Place Agent 0 at (2, 1) [adjacent only to Coop Food ID_3]
    # Place Agent 1 at (4, 4) [far away]
    env.players[0].position = (2, 1)
    env.players[1].position = (4, 4)

    # Scenario A: Solo on Coop Food (Level 3 requires level sum >= 3, Agent 0 only has level 2)
    obs, rewards, term, trunc, info = env.step([5, 0])
    print(f"Scenario A (Solo on Coop): Agent 0 Reward = {rewards[0]} (Expected: [0. 0. 0.])")

    # Scenario B: Both agents load Coop Food (2 + 2 = 4 >= 3)
    env.players[0].position = (2, 1)
    env.players[1].position = (2, 3)
    obs, rewards, term, trunc, info = env.step([5, 5])
    print(f"Scenario B (Joint on Coop): Agent 0 = {rewards[0]}, Agent 1 = {rewards[1]} (Expected: [0. 0. 1.])")

    # Scenario C: Agent 0 solo loads Food 1 (ID_2, Level 2) at (0, 1)
    env.players[0].position = (0, 1)
    obs, rewards, term, trunc, info = env.step([5, 0])
    print(f"Scenario C (Solo on Low Food): Agent 0 Reward = {rewards[0]} (Expected: [0. 0.5 0.])")

    # -------------------------------------------------------------
    # TEST 3: Stochastic Food (ID_1, Level 1) Distribution Test
    # -------------------------------------------------------------
    print("\n--- TEST 3: Stochastic Food (ID_1) Distribution Test ---")
    trials = 100
    payouts = []

    for _ in range(trials):
        env.reset()
        env.field.fill(0)

        # Place Food 0 (ID_1, Level 1) at (3, 3)
        env.food_items[0]["pos"] = (3, 3)
        env.food_items[0]["active"] = True
        env.field[3, 3] = 1

        env.food_items[1]["active"] = False
        env.food_items[2]["active"] = False

        env.players[0].position = (3, 2)
        env.players[1].position = (0, 0)

        obs, rewards, term, trunc, info = env.step([5, 0])
        r_vec = rewards[0]

        assert r_vec[1] == 0.0 and r_vec[2] == 0.0, f"Spurious reward in other channels: {r_vec}"
        assert r_vec[0] in [0.0, 1.0], f"Unexpected reward value: {r_vec[0]}"
        payouts.append(r_vec[0])

    count_0 = payouts.count(0.0)
    count_1 = payouts.count(1.0)
    empirical_mean = np.mean(payouts)

    print(f"Empirical Distribution over {trials} harvests:")
    print(f"  Count r = 0.0: {count_0:2d} ({count_0 / trials * 100:.1f}%)")
    print(f"  Count r = 1.0: {count_1:2d} ({count_1 / trials * 100:.1f}%)")
    print(f"  Empirical Mean Payoff: {empirical_mean:.2f} (Theoretical: 0.50)")
    assert 0.35 <= empirical_mean <= 0.65
    print(">> All tests PASSED!")