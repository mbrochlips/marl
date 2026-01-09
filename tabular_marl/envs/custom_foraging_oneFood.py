import lbforaging
from lbforaging.foraging.environment import ForagingEnv


class CustomForagingOneFood(ForagingEnv):
    def __init__(
        self,
        *args,
        pos_players=None,
        pos_foods=None,
        **kwargs
    ):
        # Call the parent constructor with all original arguments
        super().__init__(*args, **kwargs)

        # Store your custom arguments
        self.pos_players = pos_players
        self.pos_foods = pos_foods

    def spawn_players(self, min_player_levels, max_player_levels):
        
        levels = [min_player_levels[0], max_player_levels[0]]
        for i in range(len(self.players)):
            self.players[i].reward = 0
            self.players[i].setup(
                    (self.pos_players[i][0],self.pos_players[i][1]),
                    levels[i],
                    self.field_size,
            )       

    def spawn_food(self, max_num_food, min_levels, max_levels):
        min_levels = max_levels if self.force_coop else min_levels
        levels = [min_levels[0], max_levels[0]]

        for i in range(max_num_food):
            self.field[self.pos_foods[i][0],self.pos_foods[i][1]] = levels[(i+1)%2]
        self._food_spawned = self.field.sum()


    def step(self, actions):
        # Store food sum before step
        food_before = self.field.sum()
        
        # Call parent step (handles all game logic correctly)
        obs, rewards, done, truncated, info = super().step(actions)
        
        # End episode if ANY food was collected
        if self.field.sum() < food_before:
            done = True
            self._game_over = True
        
        return obs, rewards, done, truncated, info