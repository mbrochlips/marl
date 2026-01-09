import numpy as np
import gymnasium as gym


class MoveChairCoordination(gym.Env):
    def __init__(self, ep_length, step_cost=0.01, reward_shaping=False):
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

        # Actions:
        # 0: noop (hold door if at pos 2 without chair, enter closet if at pos 2 with chair)
        # 1: left (exit closet to pos 2 if at pos 4)
        # 2: right (exit closet to pos 2 if at pos 4)
        # 3: pickup/putdown
        self.door_reset_timer = 0
        # Observation space: each agent observes [self_Pos, self_HasChair, opp_Pos, opp_HasChair, ChairPos, DoorOpen]
        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Tuple([
                gym.spaces.Discrete(5),  # self_pos: 0, 1, 2, 3, 4
                gym.spaces.Discrete(2),  # self_has_chair: 0, 1
                gym.spaces.Discrete(5),  # opp_pos: 0, 1, 2, 3, 4
                gym.spaces.Discrete(2),  # opp_has_chair: 0, 1
                gym.spaces.Discrete(5),  # chair_pos: 0, 1, 2, 3, 4
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
        spawn_poss = [0,1,2,3]
        A1_pos = np.random.choice(spawn_poss)
        spawn_poss.pop(A1_pos)
        A2_pos = np.random.choice(spawn_poss)
        self._state[:] = [1, 0, 1, 0, 0, 0]
        return self._get_obs(), {}
    
    def _is_terminal(self):
        """Check if current state matches terminal condition.
        Terminal: one agent at door(2) with chair, other at door holding it open.
        """
        s = self._state
        # chair at door (pos 2), door open
        if s[4] == 3 and s[5] == 1:
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

        # Extract current state into readable variables
        pos = [self._state[0], self._state[2]]
        has_chair = [self._state[1], self._state[3]]
        chair_pos = self._state[4]
        door_open = self._state[5]

        # Phase 1: Compute each agent's desired action independently
        desired_pos = [pos[0], pos[1]]
        desired_has_chair = [has_chair[0], has_chair[1]]
        try_pickup = [False, False]
        will_open_door = [False, False]

        for i, a in enumerate(actions):
            if a == 0:  # noop / hold door / enter closet
                if pos[i] == 2 and has_chair[i] == 0:
                    will_open_door[i] = True
                elif pos[i] == 2 and has_chair[i] == 1:
                    # At door with chair - enter closet to get out of the way
                    desired_pos[i] = 4
                # else: true noop

            elif a == 1:  # move left
                if pos[i] == 4:
                    desired_pos[i] = 2  # exit closet to door position
                elif pos[i] > 0:
                    desired_pos[i] = pos[i] - 1

            elif a == 2:  # move right
                if pos[i] == 4:
                    desired_pos[i] = 2  # exit closet to door position
                elif pos[i] < 3:
                    desired_pos[i] = pos[i] + 1

            elif a == 3:  # pickup / putdown
                if has_chair[i] == 1:
                    desired_has_chair[i] = 0  # put down chair
                elif pos[i] == chair_pos and has_chair[1-i] == 0:
                    # Can only pick up if at chair position and other agent doesn't have it
                    try_pickup[i] = True

        # Phase 2: Resolve conflicts between agents
        new_pos = [desired_pos[0], desired_pos[1]]
        new_has_chair = [desired_has_chair[0], desired_has_chair[1]]

        # Check for movement conflicts
        for i in range(2):
            other = 1 - i
            moving = desired_pos[i] != pos[i]
            
            if not moving:
                new_pos[i] = pos[i]  # not moving, stays in place
                continue

            # Check if any agent has a chair - this determines collision rules
            any_has_chair = has_chair[i] or has_chair[other]
            
            if any_has_chair:
                # With a chair involved, stricter collision rules apply
                
                # Other agent is staying at my destination (not moving away)
                other_staying_at_dest = (
                    pos[other] == desired_pos[i] and 
                    desired_pos[other] == pos[other]
                )
                
                # Both agents actively moving to the same destination
                other_moving = desired_pos[other] != pos[other]
                same_destination = other_moving and (desired_pos[i] == desired_pos[other])
                
                # Swap: both trying to move into each other's current position
                trying_to_swap = (
                    desired_pos[i] == pos[other] and 
                    desired_pos[other] == pos[i]
                )
                
                # Collision cases when chair is involved:
                # 1. Other staying at my destination (can't squeeze past with chair)
                # 2. Both moving to same spot (position conflict)
                # 3. Trying to swap positions (collide in middle)
                blocked = other_staying_at_dest or same_destination or trying_to_swap
            else:
                # Neither has chair - no collision restrictions
                # Agents can coexist at same position
                blocked = False

            if blocked:
                new_pos[i] = pos[i]  # movement fails, stay in place

        # Phase 3: Apply valid actions and update state
        new_chair_pos = chair_pos
        
        # Update chair position based on movement of agent carrying it
        for i in range(2):
            if has_chair[i] and new_pos[i] != pos[i]:
                new_chair_pos = new_pos[i]
        
        # Handle chair putdown
        for i in range(2):
            if has_chair[i] and desired_has_chair[i] == 0:
                new_chair_pos = new_pos[i]
                new_has_chair[i] = 0

        # Handle chair pickup (only one can succeed if both try)
        if try_pickup[0] and not try_pickup[1]:
            new_has_chair[0] = 1
        elif try_pickup[1] and not try_pickup[0]:
            new_has_chair[1] = 1
        # If both try, neither gets it

        # Update door state
        new_door_open = door_open
        if will_open_door[0] or will_open_door[1]:
            new_door_open = 1

        # Write back to internal state
        self._state[0] = new_pos[0]
        self._state[1] = new_has_chair[0]
        self._state[2] = new_pos[1]
        self._state[3] = new_has_chair[1]
        self._state[4] = new_chair_pos
        self._state[5] = new_door_open

        # Check termination and compute rewards
        if self._is_terminal():
            rewards = [r + 0.5 for r in rewards]
            done = True
        elif self.t >= self.ep_length:
            done = True
        else:
            done = False

        # Reward shaping: simple potential-based shaping
        if self.reward_shaping and not done:
            agents_holding_door = [
                new_pos[0] == 2 and new_has_chair[0] == 0,
                new_pos[1] == 2 and new_has_chair[1] == 0
            ]

            for i in range(self.n_agents):
                if new_has_chair[i] and not prev_has_chair[i]:
                    rewards[i] += 0.05

                if new_has_chair[i]:
                    chair_progress = (new_chair_pos - prev_chair_pos) * 0.05
                    rewards[i] += chair_progress

                if agents_holding_door[i] and sum(agents_holding_door) > 0:
                    rewards[i] += 0.02 / sum(agents_holding_door)

        return self._get_obs(), rewards, done, False, {}

    def render(self):
        print(f"State: {self._get_obs()}")
        print(f"Step {self.t} - Last actions: {self.last_actions}")
        print("-----------------")
