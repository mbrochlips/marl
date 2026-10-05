import numpy as np
import gymnasium as gym
from lbforaging.foraging.environment import Action
from lbf_decomp_r import ForagingDecompReward 


class Foraging3Foods(ForagingDecompReward):
    """
    Level-Based Foraging testbed with 3 explicit semantic reward channels:
      Channel 0 ('ID_1'): High-level food (level 2, requires both agents, r = 1.0)
      Channel 1 ('ID_2'): Low-level food (level 1, solo, r = 0.5)
      Channel 2 ('ID_3'): Stochastic low-level food (level 1, solo, r in {0.0, 1.0})
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
        # Enforce 2 players and 3 foods with players fixed at level 1
        super().__init__(
            players=2,
            min_player_level=1,
            max_player_level=1,
            min_food_level=1,
            max_food_level=2,
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
        """
        Spawns the 3 specific semantic food types at uniform random
        empty positions on the grid. Accepts *args and **kwargs to absorb 
        'min_levels' and other parameters passed by lbforaging.
        """
        self.food_items = []

        # Find all cells not currently occupied by the spawned players
        player_positions = {p.position for p in self.players}
        available_coords = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in player_positions
        ]

        # Uniformly sample 3 distinct coordinates at random
        chosen_indices = self.np_random.choice(
            len(available_coords), size=3, replace=False
        )
        coords = [available_coords[i] for i in chosen_indices]

        # Define the exact specification for each food channel
        food_specs = [
            {
                "id": 0,
                "name": "ID_1",
                "desc": "High-level Coop",
                "level": 2,
                "reward_val": 1.0,
                "stochastic": False,
            },
            {
                "id": 1,
                "name": "ID_2",
                "desc": "Low-level Solo",
                "level": 1,
                "reward_val": 0.5,
                "stochastic": False,
            },
            {
                "id": 2,
                "name": "ID_3",
                "desc": "Stochastic Solo",
                "level": 1,
                "reward_val": None,  # Sampled on pickup
                "stochastic": True,
            },
        ]

        for spec, (r, c) in zip(food_specs, coords):
            self.field[r, c] = spec["level"]
            self.food_items.append({
                "id": spec["id"],
                "name": spec["name"],
                "desc": spec["desc"],
                "pos": (r, c),
                "level": spec["level"],
                "reward_val": spec["reward_val"],
                "stochastic": spec["stochastic"],
                "active": True,
            })

    def reset(self, **kwargs):
        """
        Resets environment and places agents and food items at fresh random locations.
        """
        obs, info = super().reset(**kwargs)
        
        # Add food metadata to info dict for diagnostics
        info["food_positions"] = {f["name"]: f["pos"] for f in self.food_items}
        return obs, info

    def step(self, actions):
        """
        Executes actions and assigns semantic rewards per channel:
          - ID_1: 1.0 if joint sum of levels >= 2 (requires both agents)
          - ID_2: 0.5 for solo or joint loading
          - ID_3: Bernoulli(0.5) in {0.0, 1.0} for solo or joint loading
        """
        self.current_step += 1

        # -------------------------------------------------------------
        # 1. Action Validation
        # -------------------------------------------------------------
        self._gen_valid_moves()
        actions = [Action(a) for a in actions]
        actions = [
            a if a in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # -------------------------------------------------------------
        # 2. Movement & Collision Resolution
        # -------------------------------------------------------------
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
            else:  # Action.NONE or Action.LOAD
                proposed_positions.append((r, c))

        new_positions = list(proposed_positions)
        collision_resolved = False

        while not collision_resolved:
            collision_resolved = True
            for i in range(len(self.players)):
                if new_positions[i] == curr_positions[i]:
                    continue

                nr, nc = new_positions[i]

                # Out of bounds or moving into food
                if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.field[nr, nc] > 0:
                    new_positions[i] = curr_positions[i]
                    collision_resolved = False
                    continue

                for j in range(len(self.players)):
                    if i != j:
                        # Same target cell collision
                        if new_positions[i] == new_positions[j]:
                            new_positions[i] = curr_positions[i]
                            new_positions[j] = curr_positions[j]
                            collision_resolved = False
                        # Edge swap collision
                        elif (new_positions[i] == curr_positions[j] and 
                              new_positions[j] == curr_positions[i]):
                            new_positions[i] = curr_positions[i]
                            new_positions[j] = curr_positions[j]
                            collision_resolved = False
                        # Bumping into stationary agent
                        elif new_positions[i] == curr_positions[j] and new_positions[j] == curr_positions[j]:
                            new_positions[i] = curr_positions[i]
                            collision_resolved = False

        for p, pos, a in zip(self.players, new_positions, actions):
            p.position = pos
            p.history.append(a)

        # -------------------------------------------------------------
        # 3. Semantic Reward Assignment
        # -------------------------------------------------------------
        # Initialize [0.0, 0.0, 0.0] for both agents
        rewards = [np.zeros(3, dtype=np.float32) for _ in range(len(self.players))]

        for food in self.food_items:
            fr, fc = food["pos"]
            if not food["active"]:
                continue
            
            food_level = food["level"]

            # Adjacent players executing LOAD
            adj_players = self.adjacent_players(fr, fc)
            loading_players = [
                p for p in adj_players
                if actions[self.players.index(p)] == Action.LOAD
            ]

            total_level = sum(p.level for p in loading_players)

            # Check if loading threshold is met
            if loading_players and total_level >= food_level:
                food["active"] = False
                self.field[fr, fc] = 0  # Remove food from grid

                # Determine channel reward value
                if food["id"] == 0:
                    # High-Level Coop: r = 1.0 (requires both players since total_level >= 2)
                    payout = 1.0
                elif food["id"] == 1:
                    # Deterministic Solo: r = 0.5
                    payout = 0.5
                elif food["id"] == 2:
                    # Stochastic Solo: r in {0.0, 1.0} with 50% probability
                    payout = float(self.np_random.choice([0.0, 1.0]))

                # Assign payout to all participants in this food's dedicated channel
                for p in loading_players:
                    p_idx = self.players.index(p)
                    rewards[p_idx][food["id"]] = payout

        # -------------------------------------------------------------
        # 4. State Update & Output
        # -------------------------------------------------------------
        for p_idx, p in enumerate(self.players):
            p.reward_vec = rewards[p_idx]
            p.reward = float(rewards[p_idx].sum())
            p.score += p.reward

        terminated = bool(self.field.sum() == 0)
        truncated = bool(self.current_step >= self._max_episode_steps and not terminated)

        self._gen_valid_moves()
        obs = self._make_gym_obs()

        info = self._get_info()
        info["reward_vec"] = [r.copy() for r in rewards]
        info["channel_names"] = self.REWARD_CHANNELS
        info["active_foods"] = {f["name"]: f["active"] for f in self.food_items}

        return obs, rewards, terminated, truncated, info

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
    # TEST 2: Deterministic Mechanics Test
    # -------------------------------------------------------------
    print("\n--- TEST 2: Deterministic Mechanics Test ---")
    env.reset()

    # Clear field and explicitly place ALL 3 foods in known, isolated positions
    env.field.fill(0)

    # Food 0 (High Coop, level 2) at (2, 2)
    env.food_items[0]["pos"] = (2, 2)
    env.food_items[0]["active"] = True
    env.field[2, 2] = 2

    # Food 1 (Solo, level 1) far away at (0, 0)
    env.food_items[1]["pos"] = (0, 0)
    env.food_items[1]["active"] = True
    env.field[0, 0] = 1

    # Food 2 (Stochastic Solo, level 1) far away at (5, 5)
    env.food_items[2]["pos"] = (5, 5)
    env.food_items[2]["active"] = True
    env.field[5, 5] = 1

    # Place Agent 0 at (2, 1) [adjacent ONLY to Coop Food 0]
    # Place Agent 1 at (4, 4) [far away from Coop Food 0]
    env.players[0].position = (2, 1)
    env.players[1].position = (4, 4)

    # Scenario A: Agent 0 tries solo load on Coop Food (Action 5 = LOAD, Action 0 = NONE)
    obs, rewards, term, trunc, info = env.step([5, 0])
    print(f"Scenario A (Solo on Coop): Agent 0 Reward = {rewards[0]} (Expected: [0. 0. 0.])")

    # Scenario B: Agent 1 moves adjacent to (2, 2) at (2, 3), both LOAD together
    env.players[0].position = (2, 1)
    env.players[1].position = (2, 3)
    obs, rewards, term, trunc, info = env.step([5, 5])
    print(f"Scenario B (Joint on Coop): Agent 0 = {rewards[0]}, Agent 1 = {rewards[1]} (Expected: [1. 0. 0.])")

    # Scenario C: Agent 0 moves adjacent to Solo Food 1 at (0, 1) and LOADs solo
    env.players[0].position = (0, 1)
    obs, rewards, term, trunc, info = env.step([5, 0])
    print(f"Scenario C (Solo on Low Food): Agent 0 Reward = {rewards[0]} (Expected: [0. 0.5 0.])")

    # -------------------------------------------------------------
    # TEST 3: Stochastic Food (ID_3) Mechanics & Distribution Test
    # -------------------------------------------------------------
    print("\n--- TEST 3: Stochastic Food (ID_3) Distribution Test ---")

    # 1. Single-step demonstration
    env.reset()
    env.field.fill(0)

    # Place only Food 2 (Stochastic, level 1) at (3, 3)
    env.food_items[2]["pos"] = (3, 3)
    env.food_items[2]["active"] = True
    env.field[3, 3] = 1

    # Deactivate foods 0 and 1 for isolation
    env.food_items[0]["active"] = False
    env.food_items[1]["active"] = False

    # Place Agent 0 at (3, 2) [adjacent to Food 2] and Agent 1 far away at (0, 0)
    env.players[0].position = (3, 2)
    env.players[1].position = (0, 0)

    obs, rewards, term, trunc, info = env.step([5, 0])  # Action 5 = LOAD
    print(f"Single harvest sample: Agent 0 Reward = {rewards[0]} (Should be either [0. 0. 1.] or [0. 0. 0.])")

    # 2. Run 100 trials to verify the Bernoulli(0.5) distribution
    trials = 100
    payouts = []

    for _ in range(trials):
        env.reset()
        env.field.fill(0)

        env.food_items[2]["pos"] = (3, 3)
        env.food_items[2]["active"] = True
        env.field[3, 3] = 1

        env.food_items[0]["active"] = False
        env.food_items[1]["active"] = False

        env.players[0].position = (3, 2)
        env.players[1].position = (0, 0)

        obs, rewards, term, trunc, info = env.step([5, 0])
        r_vec = rewards[0]

        # Assert channels 0 and 1 remain completely unaffected
        assert r_vec[0] == 0.0 and r_vec[1] == 0.0, f"Error: Leakage into other channels: {r_vec}"
        # Assert payout is strictly binary
        assert r_vec[2] in [0.0, 1.0], f"Error: Unexpected payout value: {r_vec[2]}"

        payouts.append(r_vec[2])

    count_0 = payouts.count(0.0)
    count_1 = payouts.count(1.0)
    empirical_mean = np.mean(payouts)

    print(f"\nEmpirical Distribution over {trials} harvests:")
    print(f"  Count r = 0.0: {count_0:2d} ({count_0 / trials * 100:.1f}%)")
    print(f"  Count r = 1.0: {count_1:2d} ({count_1 / trials * 100:.1f}%)")
    print(f"  Empirical Mean Payoff: {empirical_mean:.2f} (Theoretical Expectation: 0.50)")

    # Assert statistical validity (within 3 standard deviations for N=100)
    assert 0.35 <= empirical_mean <= 0.65, f"Distribution mean {empirical_mean} deviated too far from 0.5!"
    print(">> Stochastic Food test PASSED! Channel ID_3 behaves as a true Bernoulli(0.5).")