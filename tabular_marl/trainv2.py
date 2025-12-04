import copy
import random
import os

import gymnasium as gym
import numpy as np

from random_agent import Random
from iql import IQL
from jal import JAL

from utils import (
    visualise_q_tables,
    visualise_q_convergence,
    visualise_evaluation_returns,
)
from matrix_game import create_pd_game
from custom_foraging_env import CustomForagingEnv
dirpath = "tabular_marl/"

from video import VideoRecorder


CONFIG = {
    "runname": '4dec2025',
    "save": True,
    "algorithm": Random, # how the agents learn
    "seed": None,
    "gamma": 0.99,
    "total_eps": 20, # total episodes
    "ep_length": 25, # how long each episode is (max step for env)
    "eval_freq": 20, # how often it is evaluated
    "lr": 0.5,
    "init_epsilon": 0.9,
    "eval_epsilon": 0.05,
    "food_pos": [[1,1],[3,3]],
    "player_pos": [[0,4],[4,0]]
}
CONFIG["runname"] = f"{CONFIG['algorithm'].__name__}_{CONFIG['total_eps']}eps_{CONFIG['ep_length']}epL_{CONFIG['runname']}"

def iql_eval(env, config, q_tables, eval_episodes=500, output=True):
    """
    Evaluate configuration of independent Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_tables (List[Dict[Act, float]]): Q-tables mapping actions to Q-values for each agent
    :param eval_episodes (int): number of evaluation episodes
    :param output (bool): flag whether mean evaluation performance should be printed
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agents = config["algorithm"](
        num_agents=len(config["player_pos"]),
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["eval_epsilon"],
    )
    eval_agents.q_tables = q_tables

    episodic_returns = []
    for _ in range(eval_episodes):
        obss, _ = env.reset()
        episodic_return = np.zeros(len(config["player_pos"]))
        done = False

        while not done:
            actions = eval_agents.act(obss)
            obss, rewards, done, _, _ = env.step(actions)
            episodic_return += rewards

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns, axis=0)
    std_return = np.std(episodic_returns, axis=0)

    if output:
        print("EVALUATION RETURNS:")
        print(f"\tAgent 1: {mean_return[0]:.2f} ± {std_return[0]:.2f}")
        print(f"\tAgent 2: {mean_return[1]:.2f} ± {std_return[1]:.2f}")
    return mean_return, std_return


def train(env, config, output=True):
    """
    Train and evaluate independent Q-learning in env with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag if mean evaluation results should be printed
    :return (List[List[float]], List[List[float]], List[Dict[Act, float]]):
    """
    agents = config["algorithm"](
        num_agents=len(config["player_pos"]),
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["init_epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["ep_length"]

    video = VideoRecorder()

    evaluation_return_means = []
    evaluation_return_stds = []
    evaluation_q_tables = []

    all_rewards = 0

    for eps_num in range(1,1+config["total_eps"]):
        obss, _ = env.reset()
        episodic_return = np.zeros(len(config["player_pos"]))
        done = False
        
        while not done:
            agents.schedule_hyperparameters(step_counter, max_steps)
            acts = agents.act(obss)
            n_obss, rewards, done, _, _ = env.step(acts)
            agents.learn(obss, acts, rewards, n_obss, done)

            all_rewards += sum(rewards)

            step_counter += 1
            episodic_return += rewards
            obss = n_obss
            
            video.record_frame(env)

        if eps_num > 0 and (eps_num) % config["eval_freq"] == 0:
            video.save(f"{dirpath}output/{config['runname']}/video/run-{eps_num}.mp4")
            video.reset()

            mean_return, std_return = iql_eval(
                env, config, agents.q_tables, output=output
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            evaluation_q_tables.append(copy.deepcopy(agents.q_tables))
        else:
            video.reset()
    print(all_rewards)
    print(agents.q_tables[0].values())
    return (
        evaluation_return_means,
        evaluation_return_stds,
        evaluation_q_tables,
        agents.q_tables[0], # only for one agent
    )


if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    # env = create_pd_game()
    # env = gym.make("lbforaging:Foraging-5x5-2p-1f-v3")

    env = CustomForagingEnv(
        field_size=(5, 5),  
        players=len(CONFIG["player_pos"]),       
        min_player_level=2,
        max_player_level=3,
        min_food_level=2,
        max_food_level=4,
        max_num_food=len(CONFIG["food_pos"]),  
        sight=5,         
        max_episode_steps=CONFIG["ep_length"],  
        force_coop=False,
        pos_foods=CONFIG["food_pos"],
        pos_players=CONFIG["player_pos"],
        )   
    env.reset()

    # Create the folder if it doesn't exist
    os.makedirs(f"{dirpath}output/{CONFIG['runname']}/video", exist_ok=True)

    evaluation_return_means, evaluation_return_stds, eval_q_tables, q_tables = train(
        env, CONFIG
    )

    #visualise_q_tables(q_tables)
    r = visualise_evaluation_returns(evaluation_return_means, evaluation_return_stds, CONFIG, dirpath)
    #visualise_q_convergence(eval_q_tables, env)