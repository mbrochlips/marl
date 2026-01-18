from collections import defaultdict
import random
from typing import List, DefaultDict, Optional

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim

from agent.iql import IQL


class RewardGroup:
    def __init__(self, reward: float, obs_a: str):
        self.reward = reward          # average reward for group
        self.observations = {obs_a}   # set of all obs_a belonging to group
        self.success = 1              # times reward matched expectations
        self.total = 1                # total visited
    
    def matches(self, r: float, threshold: float) -> bool:
        return abs(self.reward - r) <= threshold
    
    def add_observation(self, obs_a: str, r: float):
        self.observations.add(obs_a)
        self.reward = (self.reward * self.success + r) / (self.success + 1)
        self.success += 1
        self.total += 1
    
    def increase_total(self):
        self.total += 1

    def score(self,c):
        return self.reward * self.success / self.total + c * self.reward  / np.sqrt(self.total)
        # times c * boost (Proportional to reward, shrinks with more visits.)

class QBM(IQL):
    def __init__(self, num_agents: int, action_spaces: List[Space], gamma: float, learning_rate: float = 0.5, eps_decay=True, init_epsilon: float = 0.9, epsilon_min: float = 0.05, decay_fraction: float = 0.9, r_threshold=0.09, c_boost = 10.0, **kwargs):
        super().__init__(num_agents, action_spaces, gamma, learning_rate, eps_decay, init_epsilon, epsilon_min, decay_fraction, **kwargs)
        self.groups: List[RewardGroup] = []
        self.obs_to_group: dict = {}
        self.r_threshold = r_threshold
        self.c_boost = c_boost

    def find_matching_group(self, r: float) -> Optional[int]:
        for i, group in enumerate(self.groups):
            if group.matches(r, self.r_threshold):
                return i
        return None

    def get_if_update(self, i: int):
        if len(self.groups) <= 1:
            return True
        
        avg_score = sum(g.score(self.c_boost) for j, g in enumerate(self.groups) if j != i)
        if avg_score == 0:
            return True
        
        score_i = self.groups[i].score(self.c_boost)
        return score_i < (avg_score / (len(self.groups) - 1))

    def update_reward_model(self, r: float, obs_a: str):
        
        # obs_a belongs to a group
        if obs_a in self.obs_to_group:
            i = self.obs_to_group[obs_a]
            group = self.groups[i]
            
            if group.matches(r, self.r_threshold):
                group.add_observation(obs_a, r)
                return True
            else:
                # Reward doesn't match: +1 total and check if update
                group.increase_total()
                return self.get_if_update(i)
        
        # New observation, only track significant rewards
        if r < self.r_threshold:
            return True
        
        #Check if this reward matches an existing group
        matching_group_idx = self.find_matching_group(r)
        
        if matching_group_idx is not None:
            # Add obs to existing group
            self.groups[matching_group_idx].add_observation(obs_a, r)
            self.obs_to_group[obs_a] = matching_group_idx
        else:
            #Create new group
            new_group = RewardGroup(r, obs_a)
            self.obs_to_group[obs_a] = len(self.groups)
            self.groups.append(new_group)
        
        return True


    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
    ):
        """Updates the Q-tables based on agents' experience
        :param obss (List[np.ndarray]): list of observations for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray]): list of observations after taking the action for each agent
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (List[float]): updated Q-values for current actions of each agent
        """
        #update reward_model
        update_bool = self.update_reward_model(rewards[0], str((obss[0],actions[0])))
            
        if done:
            q_next = 0
            #for i, g in enumerate(self.groups):
            #    print(f"Group {i}: reward={g.reward:.3f}, success={g.success}, total={g.total}, obs_count={len(g.observations)}")
        
        else:
            q_values_next = [self.q_tables[0][str((n_obss[0],a))] for a in range(self.n_acts[0])]
            q_next = max(q_values_next)
            #q_next = self.q_tables[i][str((n_obss[i],a_next))]

        if update_bool:  
            self.q_tables[0][str((obss[0],actions[0]))] += self.learning_rate * (rewards[0] + self.gamma*q_next - self.q_tables[0][str((obss[0],actions[0]))])