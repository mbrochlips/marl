import inspect
import gymnasium as gym
import numpy as np
from lbforaging.foraging.environment import Action, ForagingEnv

MOVES = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.WEST: (0, -1),
    Action.EAST: (0, 1),
}


class ForagingDecompReward(ForagingEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        players=2,
        min_player_level=3,
        max_player_level=3,
        field_size=(6, 6),
        sight=6,
        max_episode_steps=50,
        force_coop=False,
        normalize_reward=False,
        step_penalty=0.01,
        collision_penalty=0.0,
        miscoordination_penalty=1.0,
        food_to_collect=None,
        vector_reward=False,
        channel_names=None,
        num_food_types=6,
        normalize_coordinates=True,
        observe_agent_levels=False,
        flatten_obs=False,
        shuffle_entities=False,
        max_food=None,
        **kwargs,
    ):
        self.food_items = []
        self.food_to_collect = food_to_collect
        self.vector_reward = vector_reward
        self.step_penalty = float(step_penalty) if step_penalty is not None else 0.0
        self.collision_penalty = float(collision_penalty) if collision_penalty is not None else 0.0
        self.miscoordination_penalty = float(miscoordination_penalty) if miscoordination_penalty is not None else 0.0

        # Channel mapping
        self.channel_names = list(channel_names) if channel_names is not None else []
        self.num_reward_channels = len(self.channel_names)
        self.channel_map = {name: i for i, name in enumerate(self.channel_names)}

        # Observation configuration
        self.num_food_types = num_food_types
        self.normalize_coordinates = normalize_coordinates
        self.observe_agent_levels = observe_agent_levels
        self.flatten_obs = flatten_obs
        self.shuffle_entities = shuffle_entities

        # Token schema: [x, y, f1..fK, is_agent, is_self] -> 2 + K + 1 + 1
        self.entity_dim = 2 + self.num_food_types + 2

        # Ensure max_food is available for observation space construction
        if max_food is not None:
            self.max_food = max_food
        elif "max_num_food" in kwargs:
            self.max_food = kwargs.pop("max_num_food")
        else:
            self.max_food = getattr(self, "max_food", self.num_food_types)
        self.max_num_food = self.max_food
        
        env_kwargs = {
            "players": players,
            "min_player_level": min_player_level,
            "max_player_level": max_player_level,
            "field_size": field_size,
            "max_food": self.max_food,
            "sight": sight,
            "max_episode_steps": max_episode_steps,
            "force_coop": force_coop,
            "normalize_reward": normalize_reward,
            **kwargs,
        }
        sig = inspect.signature(ForagingEnv.__init__)
        super().__init__(**{k: v for k, v in env_kwargs.items() if k in sig.parameters})
        
        # Defensive initialization
        self.observation_space = self._get_observation_space()
        self.episode_reward_vec = np.zeros((players, self.num_reward_channels), dtype=np.float32)
        for player in self.players:
            player.reward_vec = np.zeros(self.num_reward_channels, dtype=np.float32)

    def _get_observation_space(self):
        """Constructs an observation space bounded strictly within [0.0, 1.0]."""
        num_players = len(self.players) if hasattr(self, "players") else 2
        max_food = getattr(self, "max_food", self.num_food_types)
        num_entities = num_players + max_food

        obs_shape = (num_entities * self.entity_dim,) if self.flatten_obs else (num_entities, self.entity_dim)
        agent_space = gym.spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
        return gym.spaces.Tuple(tuple([agent_space] * num_players))

    def _normalize_position(self, pos, is_partial_obs):
        if not self.normalize_coordinates:
            return float(pos[0]), float(pos[1])

        if is_partial_obs:
            denom_r = max(1, 2 * self.sight)
            denom_c = max(1, 2 * self.sight)
        else:
            denom_r = max(1, self.field_size[0] - 1)
            denom_c = max(1, self.field_size[1] - 1)

        return pos[0] / denom_r, pos[1] / denom_c

    def _make_gym_obs(self):
        observations = []
        is_partial_obs = self.sight < max(self.field_size)
        food_positions = np.argwhere(self.field > 0)

        for player in self.players:
            entities = []

            # 1. Agents
            for other_agent in self.players:
                is_self = float(other_agent == player)

                if is_partial_obs:
                    transformed = self._transform_to_neighborhood(
                        player.position, self.sight, other_agent.position
                    )
                    visible = (min(transformed) >= 0) and (max(transformed) <= 2 * self.sight)
                    pos = transformed
                else:
                    visible = True
                    pos = other_agent.position

                if visible:
                    agent_entity = np.zeros(self.entity_dim, dtype=np.float32)
                    agent_entity[0], agent_entity[1] = self._normalize_position(pos, is_partial_obs)

                    if self.observe_agent_levels:
                        max_lvl = float(np.max(self.max_player_level))
                        agent_entity[2 + self.num_food_types] = float(np.squeeze(other_agent.level)) / max(1.0, max_lvl)
                    else:
                        agent_entity[2 + self.num_food_types] = 1.0  # is_agent

                    agent_entity[2 + self.num_food_types + 1] = is_self
                    entities.append(agent_entity)
                else:
                    entities.append(np.zeros(self.entity_dim, dtype=np.float32))

            # 2. Food Items
            visible_food_count = 0
            for r, c in food_positions:
                f_level = int(self.field[r, c])
                target_pos = (r, c)

                if is_partial_obs:
                    transformed = self._transform_to_neighborhood(
                        player.position, self.sight, target_pos
                    )
                    visible = (min(transformed) >= 0) and (max(transformed) <= 2 * self.sight)
                    pos = transformed
                else:
                    visible = True
                    pos = target_pos

                if visible:
                    food_entity = np.zeros(self.entity_dim, dtype=np.float32)
                    food_entity[0], food_entity[1] = self._normalize_position(pos, is_partial_obs)

                    if 1 <= f_level <= self.num_food_types:
                        food_entity[2 + (f_level - 1)] = 1.0

                    food_entity[2 + self.num_food_types] = 0.0      # is_agent = 0
                    food_entity[2 + self.num_food_types + 1] = 0.0  # is_self = 0
                    entities.append(food_entity)
                    visible_food_count += 1

            # 3. Padding (Empty / Eaten Food Slots)
            missing_foods = self.max_food - visible_food_count
            for _ in range(max(0, missing_foods)):
                entities.append(np.zeros(self.entity_dim, dtype=np.float32))

            # 4. Formatting & Optional Shuffling
            entities = np.array(entities, dtype=np.float32)
            if self.shuffle_entities:
                np.random.shuffle(entities)

            if self.flatten_obs:
                entities = entities.flatten()

            observations.append(entities)

        return tuple(observations)

    def spawn_food(self, *args, **kwargs):
        """Intercepts base LBF spawn_food so super().reset() invokes our custom setup directly."""
        self._setup_food_items()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. Enforce scalar int levels (e.g. 3) and initialize reward_vec on all players
        for i, player in enumerate(self.players):
            if hasattr(self.max_player_level, "__getitem__"):
                player.level = int(self.max_player_level[i])
            else:
                player.level = int(self.max_player_level)
            player.reward_vec = np.zeros(self.num_reward_channels, dtype=np.float32)

        self.episode_reward_vec = np.zeros((len(self.players), self.num_reward_channels), dtype=np.float32)
        self._game_over = False
        self._gen_valid_moves()

        # 2. Return fresh zero reward vectors (foolproof against any attribute missing)
        info = {
            "reward_vec": [np.zeros(self.num_reward_channels, dtype=np.float32) for _ in self.players],
            "foods_collected": {f["type"]: not f["active"] for f in self.food_items},
            "num_foods_collected": 0,
        }
        return self._make_gym_obs(), info

    def _setup_food_items(self):
        """Implemented by child class."""
        pass

    def _reward_shape(self, r):
        return r if self.vector_reward else [float(p.reward) for p in self.players]

    def _calculate_food_reward(self, food, loading_players):
        return {p: float(food["level"]) / len(loading_players) for p in loading_players}

    def _calculate_failed_load_penalty(self, food, loading_players):
        return {}

    def step(self, actions):
        self.current_step += 1
        self._gen_valid_moves()

        # Robust action translation (handles integers, Enum instances, and numpy types)
        parsed_actions = []
        for p, a in zip(self.players, actions):
            a_int = a.value if hasattr(a, "value") else int(a)
            action_enum = Action(a_int) if Action(a_int) in self._valid_actions[p] else Action.NONE
            parsed_actions.append(action_enum)
        actions = parsed_actions

        # 1. Resolve Movements & Collisions (N-Agent Resolver)
        curr_pos = [p.position for p in self.players]
        new_pos = [
            (curr_pos[i][0] + MOVES.get(a, (0, 0))[0], curr_pos[i][1] + MOVES.get(a, (0, 0))[1])
            for i, a in enumerate(actions)
        ]
        collided_players = set()
        resolved = False

        while not resolved:
            resolved = True

            # Step 1: Boundaries & Food Obstacles
            for i, (r, c) in enumerate(new_pos):
                if new_pos[i] != curr_pos[i]:
                    if not (0 <= r < self.rows and 0 <= c < self.cols) or self.field[r, c] > 0:
                        new_pos[i] = curr_pos[i]
                        resolved = False

            # Step 2: Head-on Swaps
            for i in range(len(self.players)):
                for j in range(i + 1, len(self.players)):
                    if (
                        new_pos[i] != curr_pos[i]
                        and new_pos[j] != curr_pos[j]
                        and new_pos[i] == curr_pos[j]
                        and new_pos[j] == curr_pos[i]
                    ):
                        new_pos[i], new_pos[j] = curr_pos[i], curr_pos[j]
                        collided_players.update([i, j])
                        resolved = False

            # Step 3: Same-Tile Contests
            dest_claims = {}
            for i, pos in enumerate(new_pos):
                if pos != curr_pos[i]:
                    dest_claims.setdefault(pos, []).append(i)

            for pos, claimants in dest_claims.items():
                if len(claimants) > 1:
                    for i in claimants:
                        new_pos[i] = curr_pos[i]
                        collided_players.add(i)
                    resolved = False

            # Step 4: Stationary Cascades
            stationary_cells = {curr_pos[i] for i, pos in enumerate(new_pos) if pos == curr_pos[i]}
            for i, pos in enumerate(new_pos):
                if pos != curr_pos[i] and pos in stationary_cells:
                    new_pos[i] = curr_pos[i]
                    collided_players.add(i)
                    resolved = False

        for p, pos, a in zip(self.players, new_pos, actions):
            p.position = pos
            p.history.append(a)

        # 2. Rewards Allocation
        rewards = [np.zeros(self.num_reward_channels, dtype=np.float32) for _ in self.players]

        col_ch = self.channel_map.get("collision_penalty")
        if col_ch is not None and self.collision_penalty > 0.0:
            for p_idx in collided_players:
                rewards[p_idx][col_ch] -= abs(self.collision_penalty)

        step_ch = self.channel_map.get("step_penalty")
        if step_ch is not None and self.step_penalty > 0.0:
            for r in rewards:
                r[step_ch] -= abs(self.step_penalty)

        # 3. Food Interactions
        for food in self.food_items:
            if not food["active"]:
                continue
            fr, fc = food["pos"]
            adj_players = self.adjacent_players(fr, fc)
            loaders = [p for p in adj_players if actions[self.players.index(p)] == Action.LOAD]
            if not loaders:
                continue

            ch = self.channel_map[food["type"]]
            total_loaders_level = sum(int(np.squeeze(p.level)) for p in loaders)
            if total_loaders_level >= int(food["level"]):
                food["active"] = False
                self.field[fr, fc] = 0
                for p, val in self._calculate_food_reward(food, loaders).items():
                    rewards[self.players.index(p)][ch] += val
            else:
                for p, val in self._calculate_failed_load_penalty(food, loaders).items():
                    rewards[self.players.index(p)][ch] += val

        # 4. State & Termination
        for p_idx, p in enumerate(self.players):
            p.reward_vec = rewards[p_idx]
            p.reward = float(rewards[p_idx].sum())
            p.score += p.reward
            self.episode_reward_vec[p_idx] += rewards[p_idx]

        num_collected = sum(not f["active"] for f in self.food_items)
        target = min(self.food_to_collect, len(self.food_items)) if self.food_to_collect else len(self.food_items)
        terminated = bool(num_collected >= target or self.field.sum() == 0)
        truncated = bool(self.current_step >= self._max_episode_steps and not terminated)
        self._game_over = terminated or truncated

        self._gen_valid_moves()
        info = {
            "reward_vec": [r.copy() for r in rewards],
            "foods_collected": {f["type"]: not f["active"] for f in self.food_items},
            "num_foods_collected": num_collected,
        }
        if terminated or truncated:
            team_return = self.episode_reward_vec.sum(axis=0)
            for idx, val in enumerate(team_return):
                name = self.channel_names[idx] if self.channel_names else f"ch_{idx}"
                info[f"return_{name}"] = float(val)

        return self._make_gym_obs(), self._reward_shape(rewards), terminated, truncated, info