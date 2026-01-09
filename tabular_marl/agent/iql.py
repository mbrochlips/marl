from collections import defaultdict
import random
from typing import List, DefaultDict

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class IQL:
    """Agent using the Independent Q-Learning algorithm"""

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        eps_decay = True,
        init_epsilon: float = 0.9,
        epsilon_min: float = 0.05,  
        decay_fraction: float = 0.9, # first 90%
        **kwargs,
    ):
        """Constructor of IQL

        Initializes variables for independent Q-learning agents

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)
        :param learning_rate (float): learning rate for Q-learning updates
        :param init_epsilon (float): initial epsilon value for all agents

        :attr n_acts (List[int]): number of actions for each agent
        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents
        """
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma
        self.learning_rate = learning_rate
        
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon
        self.eps_decay = eps_decay
        self.decay_fraction = decay_fraction
        self.epsilon_min = epsilon_min
        
        # initialise Q-tables for all agents
        # access value of Q_i(o, a) with self.q_tables[i][str((o, a))] (str conversion for hashable obs)
        self.q_tables: List[DefaultDict] = [
            defaultdict(lambda: 0) for _ in range(self.num_agents)
        ]

    def act(self, obss) -> List[int]:
        """

        :param obss (List): list of observations for each agent
        :return (List[int]): index of selected action for each agent
        """
    #obss = [1,1,2] = food1_pos,level + [3,3,1] = food2_pos,level + [0,4,1] = selfpos,level + [4,0,1] = otherplayerpos,level
    # obss[i][:-3] to not include the pos of the other agent. --> include in env!
        actions = []

        for i in range(self.num_agents):
            if self.epsilon > np.random.rand():
                actions.append(random.randrange(self.n_acts[i]))
            else:
                q_values = [self.q_tables[i][str((obss[i],a))] for a in range(self.n_acts[i])]
                
                max_q = max(q_values)
                best_actions = [a for a, q in enumerate(q_values) if q == max_q]
                actions.append(random.choice(best_actions))
        
        return actions

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

        for i in range(self.num_agents):
            if done:
                q_next = 0
                #print(self.q_tables)
            else:
                q_values_next = [self.q_tables[i][str((n_obss[i],a))] for a in range(self.n_acts[i])]
                q_next = max(q_values_next)
                #q_next = self.q_tables[i][str((n_obss[i],a_next))]
                
            self.q_tables[i][str((obss[i],actions[i]))] += self.learning_rate * (rewards[i] + self.gamma*q_next - self.q_tables[i][str((obss[i],actions[i]))])

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        if self.eps_decay:
            decay_steps = self.decay_fraction * max_timestep
            self.epsilon = max(
                self.epsilon_min,
                self.init_epsilon  - (self.init_epsilon - self.epsilon_min) * (timestep / decay_steps)
            )
        else:
            self.epsilon = 0.1
        #self.epsilon = (1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99) * self.init_epsilon
        #self.epsilon = (1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99)