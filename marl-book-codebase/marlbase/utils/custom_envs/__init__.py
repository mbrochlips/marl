import gymnasium as gym
from marlbase.utils.custom_envs.foraging_nfoods import ForagingNFoods

__all__ = ["ForagingNFoods"]

ENTRY_POINT = "marlbase.utils.custom_envs.foraging_nfoods:ForagingNFoods"

gym.register(
    id="8Foods-8x8-v0",
    entry_point=ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "food_preset": "8foods",
        "players": 2,
        "min_player_level": 4,
        "max_player_level": 4,
        "field_size": (8, 8),
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
        "food_to_collect": 3,
        "vector_reward": True,
        "min_food_dist": 2,
    },
)

gym.register(
    id="4Foods-6x6-v0",
    entry_point=ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "food_preset": "4foods",
        "players": 2,
        "min_player_level": 4,
        "max_player_level": 4,
        "field_size": (6, 6),
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
        "food_to_collect": 3,
        "vector_reward": True,
        "min_food_dist": 2,
    },
)

gym.register(
    id="3Foods-6x6-v0",
    entry_point=ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "food_preset": "3foods",
        "players": 2,
        "min_player_level": 4,
        "max_player_level": 4,
        "field_size": (6, 6),
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
        "step_penalty": 0.01,
        "miscoordination_penalty": 0.2,
        "food_to_collect": 2,
        "vector_reward": True,
        "min_food_dist": 2,
    },
)