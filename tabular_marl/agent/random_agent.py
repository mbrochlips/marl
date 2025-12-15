import random
from agent.iql import IQL

class Random(IQL):

    def act(self, obss):
        return [random.randrange(self.n_acts[i]) for i in range(self.num_agents)]
    
    def learn(self, obss, actions, rewards, n_obss, done):
        None

