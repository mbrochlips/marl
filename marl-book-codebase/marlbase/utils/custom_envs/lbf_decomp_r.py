import numpy as np
import gymnasium as gym
from lbforaging.foraging.environment import ForagingEnv, Action, Player


class ForagingDecompReward(ForagingEnv):
    """
    Base level-based foraging environment with decomposed (vectorized) rewards.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_reward_channels = self.max_num_food
        self.food_items = []

    def reset(self, **kwargs):
        res = super().reset(**kwargs)
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs, info = res, {}

        # Scan grid only if food_items was not configured by a specialized subclass
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
                            "active": True,
                        })
                        food_id += 1

        for player in self.players:
            player.reward_vec = np.zeros(self.num_reward_channels, dtype=np.float32)

        self._game_over = False
        return obs, info

    def _calculate_food_reward(self, food, loading_players):
        """
        Hook for subclasses to define custom semantic reward values.
        Default: standard proportional split.
        """
        food_level = food["level"]
        total_loading_level = sum(p.level for p in loading_players)
        payouts = {}
        for p in loading_players:
            if self._normalize_reward and self._food_spawned > 0:
                val = float(food_level * (p.level / total_loading_level) / self._food_spawned)
            else:
                val = float(food_level * (p.level / total_loading_level))
            payouts[p] = val
        return payouts

    def step(self, actions):
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

        # 3. Loading & Decomposed Reward Assignment
        rewards = [
            np.zeros(self.num_reward_channels, dtype=np.float32) 
            for _ in range(len(self.players))
        ]

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

            if loading_players and total_loading_level >= food["level"]:
                food["active"] = False
                self.field[fr, fc] = 0

                # Delegate reward calculation to the hook
                payout_dict = self._calculate_food_reward(food, loading_players)
                for p, val in payout_dict.items():
                    p_idx = self.players.index(p)
                    rewards[p_idx][food["id"]] = val

        # 4. State Update & Termination
        for p_idx, p in enumerate(self.players):
            p.reward_vec = rewards[p_idx]
            p.reward = float(rewards[p_idx].sum())
            p.score += p.reward

        terminated = bool(self.field.sum() == 0)
        truncated = bool(self.current_step >= self._max_episode_steps and not terminated)
        self._game_over = bool(terminated or truncated)  # Fixes env.game_over bug

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