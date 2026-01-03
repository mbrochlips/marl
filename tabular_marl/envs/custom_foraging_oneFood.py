import lbforaging
import gymnasium as gym
from lbforaging.foraging.environment import ForagingEnv 
import numpy as np
from collections import namedtuple, defaultdict
from enum import Enum

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


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
            self.current_step += 1

            for p in self.players:
                p.reward = 0

            actions = [
                Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
                for p, a in zip(self.players, actions)
            ]

            # check if actions are valid
            for i, (player, action) in enumerate(zip(self.players, actions)):
                if action not in self._valid_actions[player]:
                    self.logger.info(
                        "{}{} attempted invalid action {}.".format(
                            player.name, player.position, action
                        )
                    )
                    actions[i] = Action.NONE

            loading_players = set()

            # move players
            # if two or more players try to move to the same location they all fail
            collisions = defaultdict(list)

            # so check for collisions
            for player, action in zip(self.players, actions):
                if action == Action.NONE:
                    collisions[player.position].append(player)
                elif action == Action.NORTH:
                    collisions[(player.position[0] - 1, player.position[1])].append(player)
                elif action == Action.SOUTH:
                    collisions[(player.position[0] + 1, player.position[1])].append(player)
                elif action == Action.WEST:
                    collisions[(player.position[0], player.position[1] - 1)].append(player)
                elif action == Action.EAST:
                    collisions[(player.position[0], player.position[1] + 1)].append(player)
                elif action == Action.LOAD:
                    collisions[player.position].append(player)
                    loading_players.add(player)

            # and do movements for non colliding players
            for k, v in collisions.items():
                if len(v) > 1:  # make sure no more than an player will arrive at location
                    continue
                v[0].position = k

            # finally process the loadings:
            while loading_players:
                # find adjacent food
                player = loading_players.pop()
                frow, fcol = self.adjacent_food_location(*player.position)
                food = self.field[frow, fcol]

                adj_players = self.adjacent_players(frow, fcol)
                adj_players = [
                    p for p in adj_players if p in loading_players or p is player
                ]

                adj_player_level = sum([a.level for a in adj_players])
                loading_players = loading_players - set(adj_players)

                if adj_player_level < food:
                    # failed to load
                    for a in adj_players:
                        a.reward -= self.penalty
                    continue

                # else the food was loaded and each player scores points
                for a in adj_players:
                    a.reward = float(a.level * food)
                    if self._normalize_reward:
                        a.reward = a.reward / float(
                            adj_player_level * self._food_spawned
                        )  # normalize reward
                # and the food is removed
                self.field[frow, fcol] = 0

            self._game_over = (
                self.field.sum() < (self.max_food_level + self.min_food_level) or self._max_episode_steps <= self.current_step #THE CHANGED LINE!
            )
            self._gen_valid_moves()

            for p in self.players:
                p.score += p.reward

            rewards = [p.reward for p in self.players]
            done = self._game_over
            truncated = False
            info = self._get_info()

            return self._make_gym_obs(), rewards, done, truncated, info