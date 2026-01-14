# agent/jal_learner.py
from collections import defaultdict
import random
from typing import List, DefaultDict
import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim
from scipy.stats import entropy
from agent.iql import IQL


class IQLAE(IQL):
    #[page 132, algorithm 8] by Albrecht and co. (marl-book)
    # only works with mixed play and 2v2!!
    """
    Joint Action Learner (JAL) - learns Q-values over joint actions Q(s, a_i, a_{-i})
    and maintains an opponent model to compute expected values.
    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        init_epsilon: float = 0.9,
        epsilon_min: float = 0.05,  
        decay_fraction: float = 0.9,
        history_length: int = 2,
        eps_decay = True,
        **kwargs,
    ):
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]
        
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon
        self.decay_fraction = decay_fraction
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay
        
        # Q-table stores Q(s, a_self, a_opponent) for the learning agent
        # Key: str((obs, my_action, opponent_action))
        # Use q_tables[0] for compatibility with MixedPlay wrapper
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0)]

        self.q_history: DefaultDict = defaultdict(list)
        self.q_history_length = history_length


    def get_unc_q(self, obs, my_action: np.ndarray = None) -> float:
    # expected Q-value for my_action given opponent model:
        q_key = str((obs, my_action))
        # Uncertainty estimation
        if len(self.q_history[q_key]) > 1:
            unc_q = (self.q_history[q_key][0] - self.q_history[q_key][1])**2
        elif len(self.q_history[q_key]) == 1:
            unc_q = self.q_history[q_key][0]**2
        else:
            unc_q = 1.0
        
        return unc_q
    
    
    def act(self, obss: List) -> List[int]:
    # for easier implementation, it only learns JAL_AM for agent[0]
    # param obss: list of observations (uses obss[0] for single agent)
        obs = obss[0]
        
        unc_qs = [self.get_unc_q(obs, a) 
                for a in range(self.n_acts[0])]
    
        if self.epsilon > np.random.rand():
            max_unc_q = max(unc_qs)
            most_unc_actions = [a for a, q in enumerate(unc_qs) if q == max_unc_q]
            action = random.choice(most_unc_actions)
        else:
            q_values = [self.q_tables[0][str((obss[0],a))] for a in range(self.n_acts[0])]
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            action = random.choice(best_actions)
        
        return [action]
    
    def learn(self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray], done: bool):
        q_key = str((obss[0],actions[0]))

        if done:
            q_next = 0
            #print(self.q_tables)
        else:
            q_values_next = [self.q_tables[0][str((n_obss[0],a))] for a in range(self.n_acts[0])]
            q_next = max(q_values_next)
        
        self.q_tables[0][q_key] += self.learning_rate * (rewards[0] + self.gamma*q_next - self.q_tables[0][q_key])

        self.q_history[q_key] += [self.q_tables[0][q_key]]
        if len(self.q_history[q_key]) > self.q_history_length:
            del self.q_history[q_key][0]