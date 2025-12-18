# agent/jal_learner.py
from collections import defaultdict
import random
from typing import List, DefaultDict
import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim


class JalAM:
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
        init_epsilon: float = 1.0,
        **kwargs,
    ):
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]
        
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon
        
        # Q-table stores Q(s, a_self, a_opponent) for the learning agent
        # Key: str((obs, my_action, opponent_action))
        self.q_table: DefaultDict = defaultdict(lambda: 0)
        
        # Opponent model: tracks action frequencies
        # opponent_counts[obs][action] = count
        self.opponent_counts: DefaultDict = defaultdict(lambda: defaultdict(int))
        self.total_opponent_obs: DefaultDict = defaultdict(int)
        
        # For compatibility with MixedPlay wrapper
        self.q_tables = [self.q_table]

    def _get_opponent_model(self, obs) -> np.ndarray:
        """Get opponent's estimated policy (probability distribution over actions)."""
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

    def _get_expected_q(self, obs, my_action) -> float:
        """Compute expected Q-value for my_action given opponent model."""
        opponent_probs = self._get_opponent_model(obs)
        expected_q = 0.0
        
        for opp_action, prob in enumerate(opponent_probs):
            q_key = str((obs, my_action, opp_action))
            expected_q += prob * self.q_table[q_key]
        
        return expected_q

    def act(self, obss: List) -> List[int]:
        """
        Select action using epsilon-greedy over expected Q-values.
        
        :param obss: list of observations (uses obss[0] for single agent)
        :return: list with single action
        """
        obs = obss[0]
        
        if self.epsilon > np.random.rand():
            # Explore
            action = random.randrange(self.n_acts[0])
        else:
            # Exploit: choose action with highest expected Q
            expected_qs = [
                self._get_expected_q(obs, a) 
                for a in range(self.n_acts[0])
            ]
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
        
        # Update opponent model
        obs_key = str(obs)
        self.opponent_counts[obs_key][opponent_action] += 1
        self.total_opponent_obs[obs_key] += 1
        
        # Q-learning update for joint action Q(s, a_me, a_opp)
        q_key = str((obs, my_action, opponent_action))
        
        if done:
            q_next = 0
        else:
            # Max expected Q over my actions in next state
            q_next = max(
                self._get_expected_q(n_obs, a) 
                for a in range(self.n_acts[0])
            )
        
        # TD update
        td_target = reward + self.gamma * q_next
        td_error = td_target - self.q_table[q_key]
        self.q_table[q_key] += self.learning_rate * td_error

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Decay epsilon over time."""
        self.epsilon = (1.0 - min(1.0, timestep / (0.8 * max_timestep)) * 0.99) * self.init_epsilon