import random
import numpy as np
from agent.iql import IQL
from gymnasium.spaces import Discrete

class pRandom(IQL):
    def __init__(self, p = 0.5, env = "mcc", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p #probability of )
        self.t_act = 0
        self.initial_obs = None

        if env == "mc":
            self.door_actions = [2,0,0,0]
            self.chair_actions = [1,3,2,2]
            # Initial observation for MoveChairGame: (pos=1, has_chair=0, opp_pos=1, opp_has_chair=0, chair_pos=0, door=0)
            self.initial_obs = (1, 0, 1, 0, 0, 0)

        elif env == "mcc":
            # Door holder: goes to pos 3, waits, comes back to 2, moves to 1 while chair exits closet, returns
            # Chair carrier: gets chair, goes to 2, enters closet, waits, exits, goes to 3
            self.door_actions = [2, 2, 0, 0, 0, 1, 1, 2, 0]
            self.chair_actions = [1, 3, 2, 2, 0, 0, 2, 2, 0]
            self.initial_obs = (1, 0, 1, 0, 0, 0)


    def act(self, obss):
        # Reset at the start of each episode (when we see initial observation)
        if self.initial_obs is not None and obss[0] == self.initial_obs:
            self.t_act = 0
        
        if self.t_act == 0:
            if self.p < np.random.rand(): # p = Change for getting chair 
                self.actions = self.door_actions
            else:
                self.actions = self.chair_actions

        if self.t_act < len(self.actions):
            a = self.actions[self.t_act]
        else:
            a = 0

        self.t_act += 1

        return [a]

    def learn(self, obss, actions, rewards, n_obss, done):
        if done:
            self.t_act = 0

