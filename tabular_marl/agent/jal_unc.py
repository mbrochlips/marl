# agent/jal_learner.py
from collections import defaultdict
import random
from typing import List, DefaultDict
import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim
from scipy.stats import entropy
from agent.iql import IQL


class JalUnc(IQL):
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
        decay_fraction: float = 0.9, # first 90%
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
        
        #opponent_counts[obs][action] = count
        self.opponent_counts: DefaultDict = defaultdict(lambda: defaultdict(int))
        
        #used for normalization 
        self.total_opponent_obs: DefaultDict = defaultdict(int)

        self.q_history: DefaultDict = defaultdict(list)
        self.q_history_length = history_length

    def opp_probs(self, obs) -> np.ndarray:
        # for easier implementation, it only works for agent at a time:
        if self.num_agents != 1:
            raise ValueError("JAL requires only one opponent!")

        # probability distribution over actions:

        obs_key = str(obs)
        total = self.total_opponent_obs[obs_key]
        
        if total == 0:
            # Uniform prior if no observations
            return np.ones(self.n_acts[0]) / self.n_acts[0]
        
        probs = np.array([
            self.opponent_counts[obs_key][a] / total 
            for a in range(self.n_acts[0])
        ])
        return probs

    def get_expected_q(self, obs, my_action, opponent_probs: np.ndarray = None) -> float:
        # expected Q-value for my_action given opponent model:
        
        if opponent_probs is None:
            opponent_probs = self.opp_probs(obs)
        
        expected_q = 0.0
        unc_q = 0.0  # Initialize outside loop to accumulate
        
        for opp_action, prob in enumerate(opponent_probs):
            q_key = str((obs, my_action, opp_action))

            #AV_i (6.17)
            expected_q += prob * self.q_tables[0][q_key]
            
            # Uncertainty estimation
            if len(self.q_history[q_key]) > 1:
                unc_q += prob * 2 * (self.q_history[q_key][0] * self.q_history[q_key][1])**2
            elif len(self.q_history[q_key]) == 1:
                unc_q += prob * self.q_history[q_key][0]
            else:
                unc_q += prob  # *1, only works with a maximum reward of 1
        
        return expected_q, unc_q

    def act(self, obss: List) -> List[int]:
        # for easier implementation, it only learns JAL_AM for agent[0]
        # param obss: list of observations (uses obss[0] for single agent)
        obs = obss[0]
        
        # Compute opponent probs ONCE for this observation
        opponent_probs = self.opp_probs(obs)
        qs = [self.get_expected_q(obs, a, opponent_probs) 
                for a in range(self.n_acts[0])]
        expected_qs = [q[0] for q in qs]
        unc_qs = [q[1] for q in qs]

        if self.epsilon > np.random.rand():
            max_unc_q = max(unc_qs)
            most_unc_actions = [a for a, q in enumerate(unc_qs) if q == max_unc_q]
            action = random.choice(most_unc_actions)
        else:
            max_q = max(expected_qs)
            best_actions = [a for a, q in enumerate(expected_qs) if q == max_q]
            action = random.choice(best_actions)
        
        return [action]

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
        opponent_action: int = None,  # Must be provided for JAL!
    ):
        """
        Update Q-table for joint action and opponent model.
        
        :param obss: observations
        :param actions: my actions
        :param rewards: my rewards  
        :param n_obss: next observations
        :param done: terminal flag
        :param opponent_action: the opponent's action (required for JAL)
        """
        obs = obss[0]
        my_action = actions[0]
        reward = rewards[0]
        n_obs = n_obss[0]
        
        
        if opponent_action is None:
            raise ValueError("JAL requires opponent_action to be provided!")
        
        # Update opponent prob model
        obs_key = str(obs)
        self.opponent_counts[obs_key][opponent_action] += 1
        self.total_opponent_obs[obs_key] += 1
        
        #Q-learning update for joint action Q(s, a_me, a_opp)
        q_key = str((obs, my_action, opponent_action))
        
        if done:
            q_next = 0
        else:
            # Max expected Q over my actions in next state (use only expected_q, not unc_q)
            # Compute opponent probs ONCE for next observation
            next_opponent_probs = self.opp_probs(n_obs)
            q_next = max(
                self.get_expected_q(n_obs, a, next_opponent_probs)[0]
                for a in range(self.n_acts[0])
            )
        
        # Q-update
        self.q_tables[0][q_key] += self.learning_rate * (reward + self.gamma * q_next - self.q_tables[0][q_key])

        # Update Q-history
        self.q_history[q_key] += [self.q_tables[0][q_key]]
        if len(self.q_history[q_key]) > self.q_history_length:
            self.q_history[q_key] = self.q_history[q_key][1:]