from collections.abc import Iterable

import numpy as np
import gymnasium as gym


class MoveChairGame(gym.Env):
    def __init__(self, ep_length, step_cost=0.05):
        """
        Create matrix game
        :param payoff_matrix: np.array of shape (n_actions_1, n_actions_2, 2)
        :param ep_length: length of episode (before done is True)
        """
        self.terminal_states = [gym.spaces.Tuple([
            [2,1,2,0,2,1],[2,0,2,1,2,1]]),
            gym.spaces.Tuple([
            [2,0,2,1,2,1],[2,1,2,0,2,1]])]

        self.step_cost = step_cost

        self.n_agents = 2
        n_actions_1, n_actions_2, _ = [4,4]
        # noop/HoldDoor, Left, Right, Pickup/PutDown

        self.observation_space = gym.spaces.Tuple([
            [gym.spaces.Discrete(3),
            gym.spaces.Discrete(2),
            gym.spaces.Discrete(3),
            gym.spaces.Discrete(2),
            gym.spaces.Discrete(3),
            gym.spaces.Discrete(2)] for _ in range(self.n_agents)])
        # self_Pos, self_HasChair, opp_Pos, opp_HasChair, ChairPos, DoorOpen

        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(n_actions_1), gym.spaces.Discrete(n_actions_2)])
        self.ep_length = ep_length
        self.last_actions = None

        self.t = 0
        self.init_state = gym.spaces.Tuple([
            [1,0,1,0,0,0] for _ in range(self.n_agents)])
        
    
    def reset(self, seed=None):
        self.t = 0
        self.state = self.init_state
        return [0] * self.n_agents, {}

    def step(self, actions):
        assert len(actions) == self.n_agents, f"Expected {self.n_agents} actions, got {len(actions)}"
        self.t += 1
        self.last_actions = actions
        rewards = [self.step_cost for _ in self.n_agents]

        #handle both reaching for chair
        try_chair = [False, False]

        common_state = self.state[0].copy()

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
                if common_state[1 + (2*i)] == 1: #put down chair
                    common_state[1 + (2*i)] = 0
                elif common_state[0 + (2*i)] == common_state[4]: #can pick up a chair
                    try_chair[i] = True
                else:
                    continue
        
        if not try_chair[0] and try_chair[1]:
            self.state[1][1] = 1
        if try_chair[0] and not try_chair[1]:
            self.state[0][1] = 1

        # use common_state to update the state
        self.state[0] = common_state
        self.state[1] = common_state[2:4] + common_state[0:2] + common_state[4:6]

        if self.state in self.terminal_states:
            rewards += 1
            done = True
        elif self.t >= self.ep_length:
            done = True
        else:
            done = False

        return self.state, rewards, done, False, {}

    def render(self):
        print(f"Step {self.t} - actions: {self.last_actions}")
