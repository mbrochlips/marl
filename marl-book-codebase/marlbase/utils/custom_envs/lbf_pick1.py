import lbforaging
from lbforaging.foraging.environment import ForagingEnv


class ForagingPickOne(ForagingEnv):
    def __init__(self, *args, **kwargs):
        # Convert our custom argument to the argument expected by ForagingEnv
        # if "time_limit" in kwargs:
        #     kwargs["max_episode_steps"] = kwargs.pop("time_limit")
        kwargs.pop("render_mode", None)

        if "max_episode_steps" not in kwargs:
            kwargs["max_episode_steps"] = 100
        
        if "field_size" in kwargs and isinstance(kwargs["field_size"], int):
            kwargs["field_size"] = (kwargs["field_size"], kwargs["field_size"])

        super().__init__(*args, **kwargs)

    def step(self, actions):
        # Store food sum before step
        food_before = self.field.sum()

        # Call parent step
        obs, rewards, done, truncated, info = super().step(actions)

        # End episode if ANY food was collected
        if self.field.sum() < food_before:
            self._game_over = True
            done = True

        return obs, rewards, done, truncated, info