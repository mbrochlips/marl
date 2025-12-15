import random
import os
from datetime import datetime

import gymnasium as gym
import numpy as np

from utils.visualizations import (
    visualise_q_tables,
    visualise_q_convergence,
    visualise_evaluation_returns,
)

from train import train_agents
from envs.matrix_game import create_matrix_game
from envs.custom_foraging_env import CustomForagingEnv
dirpath = "tabular_marl/"


CONFIG = {
    "runname": datetime.now().strftime("%d%b%Y").lower(),  # e.g."15dec2025"
    "algorithm": "IQL", # how the agents learn
    "env": "m", #game type: "f" = foraging, "cf" = costum_foraging or "m" = matrix

    "save": True,
    "visualise": False, #not working for now
    "output": True, #flag whether mean evaluation performance should be printed

    "ep_length": 100, # how long each episode is (max step for env)
    "total_eps": 10000, # total episodes
    "eval_freq": 500, # how often it is evaluated (of total_eps)
    "eval_episodes": 100, # number of episodes to run during evaluation

    "seed": None,
    "lr": 0.1, # learning rate
    "init_epsilon": 0.5,
    "eval_epsilon": 0.05,
    "num_agents": 2,
    "gamma": 0.9,

    "food_pos": [[1,1],[3,3]],
    "player_pos": [[0,4],[4,0]],
    "payoff_matrix": np.array([[[4, 4], [0, 3]], 
                               [[3, 0], [2, 2]]])
}

if CONFIG["save"]:
    CONFIG["runname"] = f"{CONFIG['total_eps']}eps_{CONFIG['ep_length']}epL_{CONFIG['runname']}_{CONFIG['algorithm']}"
    save_dir = f"{dirpath}output/{CONFIG['algorithm']}/{CONFIG['runname']}"
    CONFIG["dir"] = save_dir

    if CONFIG["env"][-1] == "f":
        CONFIG["video"] = True
        os.makedirs(f"{save_dir}/video", exist_ok=True)
    else:
        os.makedirs(f"{save_dir}", exist_ok=True)


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
            render_mode="rgb_array" if CONFIG.get("visualise") else None,
            )   
        env.reset()
    
    elif CONFIG["env"] == "f":
        env = gym.make("lbforaging:Foraging-5x5-2p-1f-v3", render_mode="rgb_array" if CONFIG.get("visualise") else None)

    elif CONFIG["env"] == "m":
        env = create_matrix_game(CONFIG["payoff_matrix"], CONFIG["ep_length"])
        CONFIG["video"] = False
        CONFIG["gamma"] = 0.0  # No bootstrapping for stateless matrix games
    
    else:
        assert "A non-valid env was chosen"
    
    
    evaluation_return_means, evaluation_return_stds, eval_q_tables, final_q_tables = train_agents(
        env, CONFIG
    )

    fig = visualise_evaluation_returns(evaluation_return_means, evaluation_return_stds, CONFIG, dirpath)
    visualise_q_tables(final_q_tables, max_states=10, show_all_actions=True)
    print(len(final_q_tables[0].values()))
    #visualise_q_convergence(eval_q_tables, env)