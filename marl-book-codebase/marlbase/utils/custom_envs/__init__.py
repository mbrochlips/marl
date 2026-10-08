from gymnasium.envs.registration import register

register(
    id="ForagingPickOne",
    entry_point="utils.custom_envs.lbf_pick1:ForagingPickOne", 
)

register(
    id="Foraging-3Foods-8x8-2p-3f-v0",
    entry_point="utils.custom_envs.lbf_3foods:Foraging3Foods",
    kwargs={
        "field_size": (8, 8),
        "max_episode_steps": 50,
        "sight": 8,
    },
)

register(
    id="Foraging-3Foods-6x6-2p-3f-v0",
    entry_point="utils.custom_envs.lbf_3foods:Foraging3Foods",
    kwargs={
        "field_size": (6, 6),
        "max_episode_steps": 50,
        "sight": 6,
    },
)