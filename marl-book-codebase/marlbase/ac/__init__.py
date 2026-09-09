from gymnasium.envs.registration import register

# Register your custom environment
register(
    id="ForagingPick1",
    entry_point="your_project_folder.your_env_file:CustomForagingEnv",
)