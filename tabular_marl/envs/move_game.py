import numpy as np
import gymnasium as gym


class MoveChairGame(gym.Env):
    def __init__(self, ep_length, step_cost=0.05, reward_shaping=False):
        """
        Create MoveChairGame environment
        :param ep_length: length of episode (before done is True)
        :param step_cost: cost per step (encourages faster solutions)
        :param reward_shaping: if True, give intermediate rewards for progress
        """
        # Terminal: agent with chair at open door --> successful exit
        # Format: [self_Pos, self_HasChair, opp_Pos, opp_HasChair, ChairPos, DoorOpen]

        self.step_cost = -step_cost
        self.reward_shaping = reward_shaping

        self.n_agents = 2
        n_actions_1, n_actions_2 = 4, 4
        # noop/HoldDoor, Left, Right, Pickup/PutDown

        # Observation space: each agent observes [self_Pos, self_HasChair, opp_Pos, opp_HasChair, ChairPos, DoorOpen]
        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Tuple([
                gym.spaces.Discrete(3),  # self_pos: 0, 1, 2
                gym.spaces.Discrete(2),  # self_has_chair: 0, 1
                gym.spaces.Discrete(3),  # opp_pos: 0, 1, 2
                gym.spaces.Discrete(2),  # opp_has_chair: 0, 1
                gym.spaces.Discrete(3),  # chair_pos: 0, 1, 2
                gym.spaces.Discrete(2),  # door_open: 0, 1
            ])
            for _ in range(self.n_agents)
        ])

        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(n_actions_1), gym.spaces.Discrete(n_actions_2)])
        self.ep_length = ep_length
        self.last_actions = None
        self.t = 0
        # Internal state as numpy array for efficient computation
        self._state = np.array([1, 0, 1, 0, 0, 0], dtype=np.int32)
        
    
    def _get_obs(self):
        """Return observations as tuples (hashable for Q-table keys)."""
        # Agent 0's view: [self_pos, self_has_chair, opp_pos, opp_has_chair, chair_pos, door_open]
        obs0 = tuple(self._state)
        # Agent 1's view: swap self/opponent perspectives
        obs1 = (self._state[2], self._state[3], self._state[0], self._state[1], 
                self._state[4], self._state[5])
        return [obs0, obs1]
    
    def reset(self, seed=None):
        self.t = 0
        spawn_poss = [0,1,2]
        A1_pos = np.random.choice(spawn_poss)
        # spawn_poss.pop(A1_pos)
        A2_pos = np.random.choice(spawn_poss)
        self._state[:] = [1, 0, 1, 0, 0, 0]
        return self._get_obs(), {}
    
    def _is_terminal(self):
        """Check if current state matches terminal condition.
        Terminal: one agent at door(2) with chair, other at door holding it open.
        """
        s = self._state
        # Both at door (pos 2), chair at door (pos 2), door open
        if s[0] == 2 and s[2] == 2 and s[4] == 2 and s[5] == 1:
            # One has chair, other doesn't
            if (s[1] == 1 and s[3] == 0) or (s[1] == 0 and s[3] == 1):
                return True
        return False

    def step(self, actions):
        assert len(actions) == self.n_agents, f"Expected {self.n_agents} actions, got {len(actions)}"
        self.t += 1
        self.last_actions = actions
        rewards = [self.step_cost for _ in range(self.n_agents)]

        # Track previous state for reward shaping
        prev_chair_pos = self._state[4]
        prev_has_chair = [self._state[1], self._state[3]]

        #handle both reaching for chair
        try_chair = [False, False]

        # Work with a copy of state as numpy array
        common_state = self._state.copy()
        common_state[5] = 0  # Reset door to closed

        #update the state:
        for i, a in enumerate(actions):

            if a == 0:
                if common_state[0 + (2*i)] == 2 and common_state[1 + (2*i)] == 0: 
                    # by the door without chair
                    common_state[5] = 1 # noop becomes hold the door.
                else:
                    continue
      
            elif a == 1: # left
                if common_state[0 + (2*i)] == 0:
                    continue # by the left wall --> noop

                if common_state[1 + (2*i)]:
                    common_state[0 + (2*i)] -= 1
                    common_state[4] -= 1
                else:
                    common_state[0 + (2*i)] -= 1

            elif a == 2: #right
                if common_state[0 + (2*i)] == 2:
                    continue # by the right wall
                if common_state[1 + (2*i)]:
                    common_state[0 + (2*i)] += 1
                    common_state[4] += 1
                else:
                    common_state[0 + (2*i)] += 1


            elif a == 3:
                if common_state[1 + (2*i)] == 1:  # put down chair
                    common_state[4] = common_state[0 + (2*i)]  # Update chair position to agent's position
                    common_state[1 + (2*i)] = 0  # Agent no longer has chair
                elif common_state[0 + (2*i)] == common_state[4]:  # can pick up a chair
                    try_chair[i] = True
                else:
                    continue
        
        # Update common_state with chair pickup (before building self._state)
        # If both try to grab, randomly choose one (or neither - current behavior is neither)
        if not try_chair[0] and try_chair[1]:
            common_state[3] = 1  # agent 1 picks up chair
        elif try_chair[0] and not try_chair[1]:
            common_state[1] = 1  # agent 0 picks up chair
        elif try_chair[0] and try_chair[1]:
            # Both try to grab - randomly assign to one agent
            if np.random.rand() < 0.5:
                common_state[1] = 1  # agent 0 gets it
            else:
                common_state[3] = 1  # agent 1 gets it

        # Update internal state
        self._state[:] = common_state

        if self._is_terminal():
            rewards = [r + 0.5 for r in rewards]
            done = True
        elif self.t >= self.ep_length:
            done = True
        else:
            done = False

        # Reward shaping: simple potential-based shaping
        # Goal: chair at position 2 AND door open
        if self.reward_shaping and not done:
            new_has_chair = [common_state[1], common_state[3]]
            # Only count as "holding door" if agent performed action 0 at door without chair
            agents_holding_door = [
                common_state[0] == 2 and common_state[1] == 0, 
                common_state[2] == 2 and common_state[3] == 0
            ]

            new_chair_pos = common_state[4]

            for i in range(self.n_agents):
                if new_has_chair[i] and not prev_has_chair[i]:
                    rewards[i] += 0.05

                if new_has_chair[i]:
                    chair_progress = (new_chair_pos - prev_chair_pos) * 0.05
                    rewards[i] += chair_progress

                if agents_holding_door[i]:
                    rewards[i] += 0.02 / sum(agents_holding_door)

        return self._get_obs(), rewards, done, False, {}

    def render(self):
        print(f"State: {self._get_obs()}")
        print(f"Step {self.t} - Last actions: {self.last_actions}")
        print("-----------------")
