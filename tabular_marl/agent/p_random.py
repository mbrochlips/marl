import random
import numpy as np
from agent.iql import IQL
from gymnasium.spaces import Discrete

class pRandom(IQL):
    def __init__(self, p = 0.5, env = "mc", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p #probability of )
        self.env = env
        self.t_act = 0

        if env == "mc":
            self.holder_actions = [2,0,0,0]
            self.chair_actions = [1,3,2,2]

    def act(self, obss):
        if self.t_act == 0:
            if self.p < np.random.rand(): # p = Change for getting chair 
                self.actions = self.holder_actions
            else:
                self.actions = self.chair_actions

        if self.t_act < len(self.actions):
            a = self.actions[self.t_act]
        else:
            a = 0

        self.t_act += 1

        return [a]

    def learn(self, obss, actions, rewards, n_obss, done):
        None

