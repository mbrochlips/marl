from collections import defaultdict
import random
from typing import List, DefaultDict, Type, Dict, Any

import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim
from agent.iql import IQL
from agent.random_agent import Random


class MixedPlay:
    """
    A wrapper class that allows two different MARL self-play algorithms 
    to play against each other.

    Only two agents.
    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        init_epsilon: float = 1.0,
        eps_decay = True,
        algorithm_1: Type = IQL,
        algorithm_2: Type = Random,
        algorithm_1_kwargs: Dict[str, Any] = None,
        algorithm_2_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon
        self.eps_decay = eps_decay
        
        #Default empty dicts for extra kwargs
        algorithm_1_kwargs = algorithm_1_kwargs or {}
        algorithm_2_kwargs = algorithm_2_kwargs or {}
        
        #Create agent 1 using algorithm_1 (controls first agent)
        self.agent_1 = algorithm_1(
            num_agents=1,
            action_spaces=[action_spaces[0]],
            gamma=gamma,
            learning_rate=learning_rate,
            init_epsilon=init_epsilon,
            eps_decay=eps_decay,
            **algorithm_1_kwargs,
            **kwargs,
        )
        
        # Create agent 2 using algorithm_2 (controls second agent)
        self.agent_2 = algorithm_2(
            num_agents=1,
            action_spaces=[action_spaces[1]] if num_agents > 1 else action_spaces,
            gamma=gamma,
            learning_rate=learning_rate,
            init_epsilon=init_epsilon,
            eps_decay=eps_decay,
            **algorithm_2_kwargs,
            **kwargs,
        )
        
        self.algorithm_1_name = algorithm_1.__name__
        self.algorithm_2_name = algorithm_2.__name__

    @property
    def q_tables(self) -> List[DefaultDict]:
        """Combined Q-tables from both agents for evaluation."""
        return self.agent_1.q_tables + self.agent_2.q_tables

    @q_tables.setter
    def q_tables(self, value: List[DefaultDict]):
        """Set Q-tables for both agents."""
        if len(value) >= 2:
            self.agent_1.q_tables = [value[0]]
            self.agent_2.q_tables = [value[1]]
        elif len(value) == 1:
            self.agent_1.q_tables = value
            self.agent_2.q_tables = [defaultdict(lambda: 0)]
    
    def copy_model_from(self, other: "MixedPlay"):
        """Copy full model state from another MixedPlay instance (including opponent models for JAL)."""
        self.q_tables = other.q_tables
        
        # Copy opponent model for JAL agents
        if hasattr(other.agent_1, 'opponent_counts'):
            self.agent_1.opponent_counts = other.agent_1.opponent_counts
            self.agent_1.total_opponent_obs = other.agent_1.total_opponent_obs
        if hasattr(other.agent_2, 'opponent_counts'):
            self.agent_2.opponent_counts = other.agent_2.opponent_counts
            self.agent_2.total_opponent_obs = other.agent_2.total_opponent_obs

    def act(self, obss: List) -> List[int]:
        # Get action from agent 1 for obs[0]
        action_1 = self.agent_1.act([obss[0]])
        
        # Get action from agent 2 for obs[1]
        action_2 = self.agent_2.act([obss[1]] if self.num_agents > 1 else obss)
        
        # Handle case where act returns a single int vs list
        if isinstance(action_1, list):
            action_1 = action_1[0]
        if isinstance(action_2, list):
            action_2 = action_2[0]
            
        return [action_1, action_2]

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
    ):

        learn_kwargs_1 = {
            "obss": [obss[0]],
            "actions": [actions[0]],
            "rewards": [rewards[0]],
            "n_obss": [n_obss[0]],
            "done": done,
        }
        #get opponent action if JAL
        if hasattr(self.agent_1, 'opponent_counts'):  # Check if JAL
            learn_kwargs_1["opponent_action"] = actions[1]
        self.agent_1.learn(**learn_kwargs_1)
        
        # Agent 2 learns
        learn_kwargs_2 = {
            "obss": [obss[1]],
            "actions": [actions[1]],
            "rewards": [rewards[1]],
            "n_obss": [n_obss[1]],
            "done": done,
        #get opponent action if JAL
        }
        if hasattr(self.agent_2, 'opponent_counts'):  # Check if JAL
            learn_kwargs_2["opponent_action"] = actions[0]
        self.agent_2.learn(**learn_kwargs_2)
        

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        
        self.agent_1.schedule_hyperparameters(timestep, max_timestep)
        self.agent_2.schedule_hyperparameters(timestep, max_timestep)
        
        #Update shared epsilon for logging purposes
        self.epsilon = self.agent_1.epsilon

    def __repr__(self) -> str:
        return f"MixedPlay({self.algorithm_1_name} vs {self.algorithm_2_name})"
