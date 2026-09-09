from gymnasium.envs.registration import register

register(
    id="ForagingPickOne",
    entry_point="utils.custom_envs.lbf_pick1:ForagingPickOne", 
)