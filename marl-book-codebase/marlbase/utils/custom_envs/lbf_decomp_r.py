import numpy as np
import gymnasium as gym
from itertools import product
from lbforaging.foraging.environment import ForagingEnv, Action, Player


class ForagingDecompReward(ForagingEnv):
    """
    Level-Based Foraging environment with decomposed (vectorized) rewards.
    
    Instead of returning a scalar reward per agent:
        r_i in R
    Each agent receives an N-dimensional reward vector:
        r_i = [r_food0, r_food1, ..., r_food(N-1)]
    where N = max_num_food.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Total number of reward channels corresponds to the number of food items
        self.num_reward_channels = self.max_num_food
        
        # Metadata list to track individual food items and their persistent IDs
        self.food_items = []

    def reset(self, **kwargs):
        res = super().reset(**kwargs)
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs, info = res, {}

        # Only scan field if food_items was not already populated by a subclass
        if not getattr(self, "food_items", None):
            self.food_items = []
            food_id = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.field[r, c] > 0:
                        self.food_items.append({
                            "id": food_id,
                            "pos": (r, c),
                            "level": int(self.field[r, c]),
                            "active": True
                        })
                        food_id += 1

        # Initialize vector rewards for all players to zero
        for player in self.players:
            player.reward_vec = np.zeros(self.num_reward_channels, dtype=np.float32)

        return obs, info

    def step(self, actions):
        """
        Executes a step with decomposed vector reward assignment.
        
        Returns:
            obs (tuple): Gym observation per agent.
            rewards (list[np.ndarray]): Vector of rewards per agent,
                                        each of shape (max_num_food,).
            terminated (bool): True if all food is collected.
            truncated (bool): True if max episode steps reached.
            info (dict): Environment diagnostics, including decomposed reward breakdown.
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
        
        # Iteratively cancel moves that result in collisions
        collision_resolved = False
        while not collision_resolved:
            collision_resolved = True
            for i in range(len(self.players)):
                if new_positions[i] == curr_positions[i]:
                    continue

                nr, nc = new_positions[i]

                # Boundary or food collision
                if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.field[nr, nc] > 0:
                    new_positions[i] = curr_positions[i]
                    collision_resolved = False
                    continue

                for j in range(len(self.players)):
                    if i != j:
                        # Two agents claiming the exact same target cell
                        if new_positions[i] == new_positions[j]:
                            new_positions[i] = curr_positions[i]
                            new_positions[j] = curr_positions[j]
                            collision_resolved = False
                        # Edge swap collision (agents attempting to cross paths)
                        elif (new_positions[i] == curr_positions[j] and 
                              new_positions[j] == curr_positions[i]):
                            new_positions[i] = curr_positions[i]
                            new_positions[j] = curr_positions[j]
                            collision_resolved = False
                        # Bumping into an agent that is remaining still
                        elif new_positions[i] == curr_positions[j] and new_positions[j] == curr_positions[j]:
                            new_positions[i] = curr_positions[i]
                            collision_resolved = False

        # Apply resolved positions and record history
        for p, pos, a in zip(self.players, new_positions, actions):
            p.position = pos
            p.history.append(a)

        # -------------------------------------------------------------
        # 3. Food Loading & Decomposed Reward Assignment
        # -------------------------------------------------------------
        # Initialize zero reward vector for each player: shape (num_agents, max_num_food)
        rewards = [
            np.zeros(self.num_reward_channels, dtype=np.float32) 
            for _ in range(len(self.players))
        ]

        # Evaluate loading condition for every active food
        for food in self.food_items:
            fr, fc = food["pos"]
            
            if not food["active"]:
                continue
            
            food_level = food["level"]

            # Identify adjacent players attempting to LOAD this specific food
            adj_players = self.adjacent_players(fr, fc)
            loading_players = [
                p for p in adj_players 
                if actions[self.players.index(p)] == Action.LOAD
            ]

            total_loading_level = sum(p.level for p in loading_players)

            # Check if combined levels meet or exceed food requirement
            if loading_players and total_loading_level >= food_level:
                food["active"] = False
                self.field[fr, fc] = 0  # Remove food from physical grid

                # Distribute channelized rewards exclusively to participating agents
                for p in loading_players:
                    p_idx = self.players.index(p)
                    
                    # Proportional share of the food value
                    if self._normalize_reward and self._food_spawned > 0:
                        reward_val = float(food_level * (p.level / total_loading_level) / self._food_spawned)
                    else:
                        reward_val = float(food_level * (p.level / total_loading_level))

                    # Place reward in the channel corresponding to this food's ID
                    rewards[p_idx][food["id"]] = reward_val

        # -------------------------------------------------------------
        # 4. Bookkeeping & Game Over Conditions
        # -------------------------------------------------------------
        for p_idx, p in enumerate(self.players):
            p.reward_vec = rewards[p_idx]
            p.reward = float(rewards[p_idx].sum())  # Keep scalar sum for internal tracking
            p.score += p.reward

        # Episode termination
        terminated = bool(self.field.sum() == 0)
        truncated = bool(self.current_step >= self._max_episode_steps and not terminated)

        self._gen_valid_moves()
        obs = self._make_gym_obs()

        info = self._get_info()
        info["reward_vec"] = [r.copy() for r in rewards]
        info["foods_collected"] = [not f["active"] for f in self.food_items]

        return obs, rewards, terminated, truncated, info

if __name__ == "__main__":
    import numpy as np

    # Instantiate a 2-agent, 3-food environment
    env = ForagingDecompReward(
        players=2,
        min_player_level=1,
        max_player_level=2,
        min_food_level=1,
        max_food_level=2,
        field_size=(8, 8),
        max_num_food=3,
        sight=8,
        max_episode_steps=50,
        force_coop=False,
        normalize_reward=False,
    )

    obs, info = env.reset()
    print(f"Number of active foods: {len(env.food_items)}")
    for f in env.food_items:
        print(f"  Food ID {f['id']} at {f['pos']} with level {f['level']}")

    # Step with random actions
    random_actions = env.action_space.sample()
    obs, rewards, terminated, truncated, info = env.step(random_actions)

    print("\nStep executed successfully!")
    print(f"Reward Agent 0 (type: {type(rewards[0])}, shape: {rewards[0].shape}): {rewards[0]}")
    print(f"Reward Agent 1 (type: {type(rewards[1])}, shape: {rewards[1].shape}): {rewards[1]}")