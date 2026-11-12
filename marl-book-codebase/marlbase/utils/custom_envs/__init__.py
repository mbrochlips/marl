# File: __init__.py

import gymnasium as gym
from marlbase.utils.custom_envs.foraging_nfoods import ForagingNFoods
from marlbase.utils.custom_envs.foraging_sixfoods import ForagingSixFoods

__all__ = ["ForagingNFoods", "ForagingSixFoods"]

ENTRY_nfoods = "marlbase.utils.custom_envs.foraging_nfoods:ForagingNFoods"
ENTRY_sixfoods = "marlbase.utils.custom_envs.foraging_sixfoods:ForagingSixFoods"
ENTRY_standard = "marlbase.utils.custom_envs.foraging_standard:ForagingDecompRewardStandard"

gym.register(
    id="6Foods-6x6-v0",
    entry_point=ENTRY_sixfoods,
    disable_env_checker=True,
    kwargs={
        "food_preset": "ALL",
        "players": 2,
        "min_player_level": 3,
        "max_player_level": 3,
        "min_food_level": 1,
        "max_food_level": 6,
        "max_num_food": 6,
        "field_size": (6, 6),
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
        "collision_penalty": 0.03,
        "food_to_collect": 3,
        "vector_reward": True,
        "min_food_dist": 2,
        "normalize_coordinates": True,
        "observe_agent_levels": False,
        "flatten_obs": True,
        "num_spawn_foods": 4,
    },
)


gym.register(
    id="Standard-8x8-v0",
    entry_point=ENTRY_nfoods,
    disable_env_checker=True,
    kwargs={
        "players": 2,
        "min_player_level": 2,
        "max_player_level": 3,
        "min_food_level": 1,
        "max_food_level": 4,
        "field_size": (8, 8),
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
        "collision_penalty": 0.03,
        "food_to_collect": 3,
        "vector_reward": True,
        "min_food_dist": 2,
    },
)


gym.register(
    id="8Foods-8x8-v0",
    entry_point=ENTRY_nfoods,
    disable_env_checker=True,
    kwargs={
        "food_preset": "8foods",
        "players": 2,
        "min_player_level": 4,
        "max_player_level": 4,
        "min_food_level": 1,
        "max_food_level": 8,
        "field_size": (8, 8),
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
        "collision_penalty": 0.0,
        "food_to_collect": 3,
        "vector_reward": True,
        "min_food_dist": 2,
    },
)

gym.register(
    id="4Foods-6x6-v0",
    entry_point=ENTRY_nfoods,
    disable_env_checker=True,
    kwargs={
        "food_preset": "4foods",
        "players": 2,
        "min_player_level": 4,
        "max_player_level": 4,
        "min_food_level": 1,
        "max_food_level": 8,
        "field_size": (6, 6),
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
        "collision_penalty": 0.0,
        "food_to_collect": 3,
        "vector_reward": True,
        "min_food_dist": 2,
    },
)

gym.register(
    id="3Foods-6x6-v0",
    entry_point=ENTRY_nfoods,
    disable_env_checker=True,
    kwargs={
        "food_preset": "3foods",
        "players": 2,
        "min_player_level": 4,
        "max_player_level": 4,
        "min_food_level": 1,
        "max_food_level": 8,
        "field_size": (6, 6),
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
        "collision_penalty": 0.0,
        "food_to_collect": 2,
        "vector_reward": True,
        "min_food_dist": 2,
    },
)