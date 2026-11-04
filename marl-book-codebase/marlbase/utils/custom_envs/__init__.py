import gymnasium as gym
from marlbase.utils.custom_envs.base import ForagingDecompReward
from marlbase.utils.custom_envs.foraging_nfoods import ForagingNFoods

__all__ = [
    "ForagingNFoods",
]

gym.register(
    id="4Foods-Penalty-6x6-v0",
    entry_point="utils.custom_envs:ForagingNFoods",
    disable_env_checker=True,
    kwargs={
        "food_preset": "4foods",
        "players": 2,
        "min_player_level": 2,
        "max_player_level": 2,
        "field_size": (6, 6),
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.3,
        "food_to_collect": 3,
        "vector_reward": True,
    },
)

gym.register(
    id="3Foods-Penalty-6x6-v0",
    entry_point="utils.custom_envs:ForagingNFoods",
    disable_env_checker=True,
    kwargs={
        "food_preset": "3foods",
        "players": 2,
        "min_player_level": 2,
        "max_player_level": 2,
        "field_size": (6, 6),
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
        "food_to_collect": 2,
        "vector_reward": True,
    },
)

gym.register(
    id="3Foods-6x6-v0",
    entry_point="utils.custom_envs:ForagingNFoods",
    disable_env_checker=True,
    kwargs={
        "food_preset": "3foods",
        "players": 2,
        "min_player_level": 2,
        "max_player_level": 2,
        "field_size": (6, 6),
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.0,
        "food_to_collect": 2,
        "vector_reward": True,
    },
)

gym.register(
    id="3Foods-Scalar-6x6-v0",
    entry_point="utils.custom_envs:ForagingNFoods",
    disable_env_checker=True,
    kwargs={
        "food_preset": "3foods",
        "players": 2,
        "min_player_level": 2,
        "max_player_level": 2,
        "field_size": (6, 6),
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.0,
        "food_to_collect": 2,
        "vector_reward": False,
    },
)

gym.register(
    id="3Foods-Scalar-Penalty-6x6-v0",
    entry_point="utils.custom_envs:ForagingNFoods",
    disable_env_checker=True,
    kwargs={
        "food_preset": "3foods",
        "players": 2,
        "min_player_level": 2,
        "max_player_level": 2,
        "field_size": (6, 6),
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
        "food_to_collect": 2,
        "vector_reward": False,
    },
)