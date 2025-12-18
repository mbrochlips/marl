import random
import numpy as np
from agent.iql import IQL
from gymnasium.spaces import Discrete

class Random(IQL):
    def __init__(self, p = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p #probability of for chosing action 0 (only in matrix game)

    def act(self, obss):
        if self.action_spaces[0] == Discrete(2):
            return 0 if self.p > np.random.rand() else 1
        else:
            return [random.randrange(self.n_acts[i]) for i in range(self.num_agents)]
    
    def learn(self, obss, actions, rewards, n_obss, done):
        None

