import gymnasium as gym
from marlbase.utils.custom_envs.base import ForagingDecompReward
from marlbase.utils.custom_envs.foraging_nfoods import (
    ForagingNFoods,
)
from marlbase.utils.custom_envs.presets import FOOD_PRESETS

__all__ = [
    "ForagingDecompReward",
    "ForagingNFoods",
    "Foraging3Foods",
    "Foraging4FoodsStandard",
    "FOOD_PRESETS",
]

# ==============================================================================
# 8 Foods (8x8 Grid, Preset configured)
# ==============================================================================

gym.register(
    id="8Foods-Penalty-8x8-v0",
    entry_point="utils.custom_envs:ForagingNFoods",
    disable_env_checker=True,
    kwargs={
        "food_preset": "8foods",
        "players": 2,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.3,
        "vector_reward": True,
    },
)

gym.register(
    id="8Foods-8x8-v0",
    entry_point="utils.custom_envs:ForagingNFoods",
    disable_env_checker=True,
    kwargs={
        "food_preset": "8foods",
        "players": 2,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.0,
        "vector_reward": True,
    },
)

gym.register(
    id="8Foods-Scalar-8x8-v0",
    entry_point="utils.custom_envs:ForagingNFoods",
    disable_env_checker=True,
    kwargs={
        "food_preset": "8foods",
        "players": 2,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.0,
        "vector_reward": False,
    },
)

# ==============================================================================
# 4 Foods (6x6 Grid)
# ==============================================================================

gym.register(
    id="4Foods-Penalty-6x6-v0",
    entry_point="utils.custom_envs:ForagingNFoods",
    disable_env_checker=True,
    kwargs={
        "food_preset": "4foods",
        "players": 2,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.3,
        "vector_reward": True,
    },
)

# ==============================================================================
# 3 Foods (6x6 Grid)
# ==============================================================================

gym.register(
    id="3Foods-Penalty-6x6-v0",
    entry_point="utils.custom_envs:ForagingNFoods",
    disable_env_checker=True,
    kwargs={
        "food_preset": "3foods",
        "players": 2,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
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
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.0,
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
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.0,
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
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
        "vector_reward": False,
    },
)