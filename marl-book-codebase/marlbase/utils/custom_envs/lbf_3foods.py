# File: marlbase/utils/custom_envs/lbf_3foods.py
import numpy as np
import gymnasium as gym
from lbforaging.foraging.environment import Action
from utils.custom_envs.lbf_decomp_r import ForagingDecompReward


class Foraging3Foods(ForagingDecompReward):
    """
    Configurable Decomposed LBF Environment with Strategic Coordination Dilemmas.
    
    Channels:
      - Channel 0 ('ID_1'): Level 1 -> Stochastic Solo (r in {0.0, 1.0})
      - Channel 1 ('ID_2'): Level 2 -> Low-level Solo  (r = 0.5)
      - Channel 2 ('ID_3'): Level 3 -> High-level Coop (Joint: r = +3.0 to +5.0, Solo attempt: r = -1.0 if mod_3)
    
    Default Modifications:
      - mod_1 = False: Step cost + Boosted coop reward (+5.0)
      - mod_2 = True:  Early termination on first food harvest (K=1)
      - mod_3 = True:  Miscoordination penalty (-1.0) on Channel 2 for failed solo coop load
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }
    REWARD_CHANNELS = ["ID_1", "ID_2", "ID_3"]

    def __init__(
        self,
        field_size=(8, 8),
        max_episode_steps=50,
        sight=8,
        penalty=0.0,
        render_mode=None,
        # --- Strategic Dilemma Settings ---
        mod_1: bool = True,               # Step cost + high coop reward
        mod_2: bool = True,                # Early termination on K foods
        mod_3: bool = False,                # Miscoordination penalty for solo coop attempt
        max_harvests: int = 1,             # Number of foods to trigger termination (if mod_2=True)
        miscoord_penalty: float = 1.0,     # Penalty for uncoordinated coop load (if mod_3=True)
        step_cost: float = 0.02,           # Step penalty (if mod_1=True)
        coop_base_reward: float = 5.0,     # Base reward for joint coop harvest
        coop_boosted_reward: float = 5.0,  # Boosted reward if mod_1 is True
        **kwargs,
    ):
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

        # Store modification flags
        self.mod_1 = mod_1
        self.mod_2 = mod_2
        self.mod_3 = mod_3
        self.max_harvests = max_harvests
        self.miscoord_penalty = miscoord_penalty
        self.step_cost = step_cost
        self.coop_reward = coop_boosted_reward if mod_1 else coop_base_reward

        # Dynamically allocate 4 channels if mod_1 is True, else keep 3
        if self.mod_1:
            self.REWARD_CHANNELS = ["ID_1", "ID_2", "ID_3", "step_cost"]
            self.num_reward_channels = 4
        else:
            self.REWARD_CHANNELS = ["ID_1", "ID_2", "ID_3"]
            self.num_reward_channels = 3
        
        # Runtime counters
        self.harvested_count = 0

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

        # 1-to-1 Mapping: ID 1 -> Level 1, ID 2 -> Level 2, ID 3 -> Level 3
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

    def reset(self, **kwargs):
        self.harvested_count = 0
        obs, info = super().reset(**kwargs)
        info["harvested_count"] = 0
        return obs, info

    def step(self, actions):
        """
        Executes step and evaluates mod_1, mod_2, and mod_3 rules.
        """
        self.current_step += 1

        # 1. Action Validation
        self._gen_valid_moves()
        actions = [Action(a) for a in actions]
        actions = [
            a if a in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # 2. Movement & Collision Resolution
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
            else:
                proposed_positions.append((r, c))

        new_positions = list(proposed_positions)
        collision_resolved = False
        while not collision_resolved:
            collision_resolved = True
            for i in range(len(self.players)):
                if new_positions[i] == curr_positions[i]:
                    continue

                nr, nc = new_positions[i]
                if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.field[nr, nc] > 0:
                    new_positions[i] = curr_positions[i]
                    collision_resolved = False
                    continue

                for j in range(len(self.players)):
                    if i != j:
                        if new_positions[i] == new_positions[j]:
                            new_positions[i] = curr_positions[i]
                            new_positions[j] = curr_positions[j]
                            collision_resolved = False
                        elif (new_positions[i] == curr_positions[j] and 
                              new_positions[j] == curr_positions[i]):
                            new_positions[i] = curr_positions[i]
                            new_positions[j] = curr_positions[j]
                            collision_resolved = False
                        elif new_positions[i] == curr_positions[j] and new_positions[j] == curr_positions[j]:
                            new_positions[i] = curr_positions[i]
                            collision_resolved = False

        for p, pos, a in zip(self.players, new_positions, actions):
            p.position = pos
            p.history.append(a)

        # 3. Harvest Evaluation & Channel Reward Assignment
        rewards = [
            np.zeros(self.num_reward_channels, dtype=np.float32) 
            for _ in range(len(self.players))
        ]
        step_harvests = 0

        for food in self.food_items:
            fr, fc = food["pos"]
            if not food["active"] or self.field[fr, fc] == 0:
                continue

            adj_players = self.adjacent_players(fr, fc)
            loading_players = [
                p for p in adj_players 
                if actions[self.players.index(p)] == Action.LOAD
            ]
            total_loading_level = sum(p.level for p in loading_players)

            # --- CASE A: Successful Harvest (Combined level >= Food Level) ---
            if loading_players and total_loading_level >= food["level"]:
                food["active"] = False
                self.field[fr, fc] = 0
                step_harvests += 1

                for p in loading_players:
                    p_idx = self.players.index(p)
                    if food["id"] == 0:
                        payout = float(self.np_random.choice([0.0, 1.0]))  # ID_1: Stoch Solo
                    elif food["id"] == 1:
                        payout = 0.5                                       # ID_2: Low Solo
                    elif food["id"] == 2:
                        payout = self.coop_reward                          # ID_3: High Coop (+3.0 or +5.0)
                    rewards[p_idx][food["id"]] = payout

            # --- CASE B: MOD_3 Miscoordination on Coop Food (Level 3) ---
            # If agent(s) attempted LOAD on Level 3 food, but total_level < 3 (failed joint action)
            elif self.mod_3 and food["id"] == 2 and loading_players and total_loading_level < food["level"]:
                for p in loading_players:
                    p_idx = self.players.index(p)
                    # Exertion penalty applied strictly to Channel 2 (Coop channel)
                    rewards[p_idx][2] = -self.miscoord_penalty

        self.harvested_count += step_harvests

        # --- MOD_1: Step Cost Deduction ---
        if self.mod_1:
            for p_idx in range(len(self.players)):
                rewards[p_idx][3] -= self.step_cost

        # 4. State Update & Termination Checks
        for p_idx, p in enumerate(self.players):
            p.reward_vec = rewards[p_idx]
            p.reward = float(rewards[p_idx].sum())
            p.score += p.reward

        # Termination conditions:
        # Mod 2: Terminate immediately upon reaching max_harvests (default K=1)
        if self.mod_2 and self.harvested_count >= self.max_harvests:
            terminated = True
        else:
            terminated = bool(self.field.sum() == 0)

        truncated = bool(self.current_step >= self._max_episode_steps and not terminated)
        self._game_over = bool(terminated or truncated)

        self._gen_valid_moves()
        obs = self._make_gym_obs()

        info = self._get_info()
        info["reward_vec"] = [r.copy() for r in rewards]
        info["harvested_count"] = self.harvested_count

        return obs, rewards, terminated, truncated, info


if __name__ == "__main__":
    # Test with default settings: mod_1=False, mod_2=True, mod_3=True
    env = Foraging3Foods(field_size=(6, 6), max_episode_steps=50)

    print("--- TEST: Validating Defaults (mod_1=False, mod_2=True, mod_3=True) ---")
    env.reset()
    env.field.fill(0)

    # Place Coop Food (ID_3, Level 3) at (2, 2)
    env.food_items[2]["pos"] = (2, 2)
    env.food_items[2]["active"] = True
    env.field[2, 2] = 3

    # Place Solo Food (ID_2, Level 2) at (0, 0)
    env.food_items[1]["pos"] = (0, 0)
    env.food_items[1]["active"] = True
    env.field[0, 0] = 2

    # Place Agent 0 at (2, 1) [adjacent to Coop Food]
    # Place Agent 1 at (4, 4) [far away]
    env.players[0].position = (2, 1)
    env.players[1].position = (4, 4)

    # 1. Test Mod 3: Solo attempt on Coop Food
    obs, rewards, term, trunc, info = env.step([5, 0])  # Action 5 = LOAD
    print(f"Agent 0 Solo LOAD on Coop Food: Reward = {rewards[0]}")
    print(f"  -> Miscoordination Penalty applied: {rewards[0][2] == -1.0} (Expected: True)")
    print(f"  -> Episode terminated prematurely? {term} (Expected: False)")

    # 2. Test Mod 2: Agent 1 harvests Solo Food at (0, 0) -> Immediate Termination (K=1)
    env.players[1].position = (0, 1)  # Move Agent 1 adjacent to (0, 0)
    obs, rewards, term, trunc, info = env.step([0, 5])
    print(f"\nAgent 1 harvests Solo Food: Reward = {rewards[1]}")
    print(f"  -> Harvest count: {info['harvested_count']}")
    print(f"  -> Early termination triggered (mod_2=True)? {term} (Expected: True)")