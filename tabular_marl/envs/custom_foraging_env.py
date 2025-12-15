import lbforaging
import gymnasium as gym
from lbforaging.foraging.environment import ForagingEnv 
import numpy as np


class CustomForagingEnv(ForagingEnv):
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


########### ORIGINAL CODE (incl. reset()) ##############
    # def spawn_players(self, min_player_levels, max_player_levels):
    #     # permute player levels
    #     player_permutation = self.np_random.permutation(len(self.players))
    #     min_player_levels = min_player_levels[player_permutation]
    #     max_player_levels = max_player_levels[player_permutation]
    #     for player, min_player_level, max_player_level in zip(
    #         self.players, min_player_levels, max_player_levels
    #     ):
    #         attempts = 0
    #         player.reward = 0

    #         while attempts < 1000:
    #             row = self.np_random.integers(0, self.rows)
    #             col = self.np_random.integers(0, self.cols)
    #             if self._is_empty_location(row, col):
    #                 player.setup(
    #                     (row, col),
    #                     self.np_random.integers(min_player_level, max_player_level + 1),
    #                     self.field_size,
    #                 )
    #                 break
    #             attempts += 1   

    # def spawn_food(self, max_num_food, min_levels, max_levels):
    #     food_count = 0
    #     attempts = 0
    #     min_levels = max_levels if self.force_coop else min_levels

    #     # permute food levels
    #     food_permutation = self.np_random.permutation(max_num_food)
    #     min_levels = min_levels[food_permutation]
    #     max_levels = max_levels[food_permutation]

    #     while food_count < max_num_food and attempts < 1000:
    #         attempts += 1
    #         row = self.np_random.integers(1, self.rows - 1)
    #         col = self.np_random.integers(1, self.cols - 1)

    #         # check if it has neighbors:
    #         if (
    #             self.neighborhood(row, col).sum() > 0
    #             or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
    #             or not self._is_empty_location(row, col)
    #         ):
    #             continue

    #         self.field[row, col] = (
    #             min_levels[food_count]
    #             if min_levels[food_count] == max_levels[food_count]
    #             else self.np_random.integers(
    #                 min_levels[food_count], max_levels[food_count] + 1
    #             )
    #         )
    #         food_count += 1
    #     self._food_spawned = self.field.sum()

    # def reset(self, seed=None, options=None):
    #     if seed is not None:
    #         # setting seed
    #         super().reset(seed=seed, options=options)

    #     self.field = np.zeros(self.field_size, np.int32)
    #     self.spawn_players(self.min_player_level, self.max_player_level)
    #     player_levels = sorted([player.level for player in self.players])

    #     self.spawn_food(
    #         self.max_num_food,
    #         min_levels=self.min_food_level,
    #         max_levels=self.max_food_level
    #         if self.max_food_level is not None
    #         else np.array([sum(player_levels[:3])] * self.max_num_food),
    #     )
    #     self.current_step = 0
    #     self._game_over = False
    #     self._gen_valid_moves()

    # def reset(self, seed=None, options=None):
    #     if seed is not None:
    #         # setting seed
    #         super().reset(seed=seed, options=options)

    #     self.field = np.zeros(self.field_size, np.int32)
    #     self.spawn_players(self.min_player_level, self.max_player_level)
    #     player_levels = sorted([player.level for player in self.players])

    #     self.spawn_food(
    #         self.max_num_food,
    #         min_levels=self.min_food_level,
    #         max_levels=self.max_food_level
    #         if self.max_food_level is not None
    #         else np.array([sum(player_levels[:3])] * self.max_num_food),
    #     )
    #     self.current_step = 0
    #     self._game_over = False
    #     self._gen_valid_moves()

    #     nobs = self._make_gym_obs()
    #     return nobs, self._get_info()