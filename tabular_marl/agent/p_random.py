import random
import numpy as np
from agent.iql import IQL
from gymnasium.spaces import Discrete

class pRandom(IQL):
    def __init__(self, p = 0.5, env = "cf1f", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p #probability of )
        self.t_act = 0
        self.initial_obs = None
        self.env = env

        if env == "mc":
            self.coop_actions = [2,0,0,0]
            self.defect_actions = [1,3,2,2]
            # Initial observation for MoveChairGame: (pos=1, has_chair=0, opp_pos=1, opp_has_chair=0, chair_pos=0, door=0)
            self.initial_obs = (1, 0, 1, 0, 0, 0)

        elif env == "mcc":
            # Door holder: goes to pos 3, waits, comes back to 2, moves to 1 while chair exits closet, returns
            # Chair carrier: gets chair, goes to 2, enters closet, waits, exits, goes to 3
            self.coop_actions = [2, 2, 0, 0, 0, 1, 1, 2, 0]
            self.defect_actions = [1, 3, 2, 2, 0, 0, 2, 2, 0]
            self.initial_obs = (1, 0, 1, 0, 0, 0)

        elif env == "cf1f":
            self.coop_actions = [3,3,3,5] 
            self.defect_actions = [2,3,2,4,2,5]
            self.initial_obs = (1,1,5,3,3,1,0,4,3,4,0,3)
            #obss = [1,1,5] = food1_pos,level + [3,3,1] = food2_pos,level + [0,4,3] = selfpos,level + [4,0,3] = otherplayerpos,level



    def act(self, obss):
        # Reset at the start of each episode (when we see initial observation)
        if self.initial_obs is not None and np.array_equal(obss[0], self.initial_obs):
            self.t_act = 0
        
        if self.t_act == 0:
            if self.p < np.random.rand(): # p = Change for getting chair 
                self.actions = self.coop_actions
                self.type = "Door"
            else:
                self.actions = self.defect_actions
                self.type = "Chair"

        if self.t_act < len(self.actions):
            a = self.actions[self.t_act]
        else:
            if self.env == "cf1f":
                a = 5
            elif self.type == "Door" and self.env == "mcc":
                a = 0
            else:
                a = 2


        self.t_act += 1

        return [a]

    def learn(self, obss, actions, rewards, n_obss, done):
        if done:
            self.t_act = 0

