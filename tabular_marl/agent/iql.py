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
        epsilon: float = 1.0,
        **kwargs,
    ):
        """Constructor of IQL

        Initializes variables for independent Q-learning agents

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)
        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr n_acts (List[int]): number of actions for each agent
        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents
        """
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        # access value of Q_i(o, a) with self.q_tables[i][str((o, a))] (str conversion for hashable obs)
        self.q_tables: List[DefaultDict] = [
            defaultdict(lambda: 0) for _ in range(self.num_agents)
        ]

    def act(self, obss) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **IMPLEMENT THIS FUNCTION**

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
                
                if sum(q_values) == 0:
                    actions.append(random.randrange(self.n_acts[i]))
                    # for better start, otherwise alot of no-ops
                else:
                    actions.append(np.argmax(q_values))
        
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
        self.epsilon = 1.0 - (min(1.0, timestep / (0.8 * max_timestep))) * 0.99
