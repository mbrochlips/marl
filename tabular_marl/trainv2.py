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
from matrix_game import create_stag_game
from custom_foraging_env import CustomForagingEnv
dirpath = "tabular_marl/"

from video import VideoRecorder


CONFIG = {
    "runname": '5dec2025',
    "save": True,
    "algorithm": IQL, # how the agents learn
    "env": "cf", #game type: "f" = foraging, "cf" = costum_foraging or "m" = matrix

    "ep_length": 50, # how long each episode is (max step for env)
    "total_eps": 10, # total episodes
    "eval_freq": 9, # how often it is evaluated (of total_eps)

    "seed": None,
    "lr": 0.5, # learning rate
    "init_epsilon": 0.9,
    "eval_epsilon": 0.05,
    "num_agents": 2,
    "gamma": 0.99,

    "food_pos": [[1,1],[3,3]],
    "player_pos": [[0,4],[4,0]],
    "payoff_matrix": np.array([[[3, 3], [0, 1]], 
                               [[1, 0], [1, 1]]])
}



if CONFIG["save"]:
    CONFIG["runname"] = f"{CONFIG['total_eps']}eps_{CONFIG['ep_length']}epL_{CONFIG['runname']}_{CONFIG['algorithm'].__name__}"
    save_dir = f"{dirpath}output/{CONFIG['algorithm'].__name__}/{CONFIG['runname']}"
    CONFIG["dir"] = save_dir

    if CONFIG["env"][-1] == "f":
        CONFIG["video"] = True
        os.makedirs(f"{save_dir}/video", exist_ok=True)
    else:
        os.makedirs(f"{save_dir}", exist_ok=True)
        

def eval(env, config, q_tables, eval_episodes=500, output=True):
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
        num_agents=config["num_agents"],
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
            obss = n_obss #the new observation becomes the current
            
            if config["video"]:
                video.record_frame(env)

        if eps_num > 0 and (eps_num) % config["eval_freq"] == 0:
            if config["video"]:
                video.save(f"{config['dir']}/video/run-{eps_num}.mp4")
                video.reset()

            mean_return, std_return = eval(
                env, config, agents.q_tables, output=output
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            evaluation_q_tables.append(copy.deepcopy(agents.q_tables))
        else:
            if config["video"]:
                video.reset()

    print(agents.q_tables[0].values())
    print(agents.q_tables[1].values())

    return (
        evaluation_return_means,
        evaluation_return_stds,
        evaluation_q_tables,
        agents.q_tables[0], # only for one agent
    )


if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    if CONFIG["env"] == "cf":
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
        
    
    elif CONFIG["env"] == "f":
        env = gym.make("lbforaging:Foraging-5x5-2p-1f-v3")


    elif CONFIG["env"] == "m":
        env = create_stag_game(CONFIG["payoff_matrix"], CONFIG["ep_length"])
        CONFIG["video"] = False
    
    else:
        assert "A non-valid env was chosen"
    
    
    evaluation_return_means, evaluation_return_stds, eval_q_tables, q_tables = train(
        env, CONFIG
    )

    #q = visualise_q_tables(q_tables)
    fig = visualise_evaluation_returns(evaluation_return_means, evaluation_return_stds, CONFIG, dirpath)
    #visualise_q_convergence(eval_q_tables, env)