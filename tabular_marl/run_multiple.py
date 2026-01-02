import random
import os
import copy
from datetime import datetime

import gymnasium as gym
import numpy as np

from utils.visualizations import (
    visualise_q_tables,
    visualise_evaluation_returns,
)
from utils.video import VideoRecorder

from agent.mixed_play_wrapper import MixedPlay
from agent.iql import IQL
from agent.random_agent import Random
from agent.jal import JalAM

from envs.matrix_game import create_matrix_game
from envs.custom_foraging_env import CustomForagingEnv
from envs.move_game import MoveChairGame

dirpath = "tabular_marl/"

# Available algorithms for mixed play
ALGORITHMS = {
    "Random": Random,
    "IQL": IQL,
    "JalAM": JalAM
}

CONFIG = {
    "runname": datetime.now().strftime("%d%b%Y").lower(),  # e.g. "15dec2025"
    
    # Mixed play configuration
    "algorithm_1": "IQL",   # Algorithm for agent 1
    "algorithm_2": "IQL",   # Algorithm for agent 2
    "algorithm_1_kwargs": {},  # extra kwargs for algorithm 1
    "algorithm_2_kwargs": {},  # Extra kwargs for algorithm 2

    "env": "cf",  # game type: "f" = foraging, "cf" = custom_foraging, "m" = matrix, "mc" = MoveChairGame

    "save": True,
    "visualise": False,
    "output": True,

    "repetitions": 50,  # Number of independent runs
    "ep_length": 100,
    "total_eps": 100,
    "eval_episodes": 30,

    "seed": None,
    "lr": 0.1,
    "init_epsilon": 0.7,
    "eval_epsilon": 0.05,
    "num_agents": 2,
    "gamma": 0.95,

    "food_pos": [[1, 1], [3, 3]],
    "player_pos": [[0, 4], [4, 0]],
    "payoff_matrix": np.array([[[4, 4], [0, 3]], 
                               [[3, 0], [2, 2]]])
}


def train_agents(env, config, rep_num=0):
    """
    Train agents without intermediate evaluation.

    :param env (gym.Env): environment to train on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param rep_num (int): repetition number for logging
    :return: trained agents
    """
    # Create mixed play agents
    agents = MixedPlay(
        num_agents=len(config["player_pos"]),
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        init_epsilon=config["init_epsilon"],
        algorithm_1=ALGORITHMS[config["algorithm_1"]],
        algorithm_2=ALGORITHMS[config["algorithm_2"]],
        algorithm_1_kwargs=config.get("algorithm_1_kwargs", {}),
        algorithm_2_kwargs=config.get("algorithm_2_kwargs", {}),
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["ep_length"]

    for eps_num in range(1, 1 + config["total_eps"]):
        obss, _ = env.reset()
        done = False

        while not done:
            agents.schedule_hyperparameters(step_counter, max_steps)
            acts = agents.act(obss)
            n_obss, rewards, done, _, _ = env.step(acts)
            agents.learn(obss, acts, rewards, n_obss, done)

            step_counter += 1
            obss = n_obss

        if config.get("output") and eps_num % (config["total_eps"] // 10) == 0:
            print(f"  Rep {rep_num + 1}: Episode {eps_num}/{config['total_eps']}")

    return agents


def evaluate_agents(env, config, trained_agents, rep_num=0):
    """
    Evaluate trained agents.

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict): configuration dictionary
    :param trained_agents (MixedPlay): the trained mixed play agents
    :param rep_num (int): repetition number for video naming
    :return (list): list of episodic returns per evaluation episode
    """
    num_agents = len(config["player_pos"])
    eval_episodes = config["eval_episodes"]

    video = VideoRecorder()
    
    # Create evaluation agents with low epsilon (greedy)
    eval_agents = MixedPlay(
        num_agents=num_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        init_epsilon=config["eval_epsilon"],
        algorithm_1=ALGORITHMS[config["algorithm_1"]],
        algorithm_2=ALGORITHMS[config["algorithm_2"]],
        algorithm_1_kwargs=config.get("algorithm_1_kwargs", {}),
        algorithm_2_kwargs=config.get("algorithm_2_kwargs", {}),
    )
    
    # Copy trained Q-tables to evaluation agents
    eval_agents.q_tables = trained_agents.q_tables

    episodic_returns = []
    for i in range(eval_episodes):
        obss, _ = env.reset()
        episodic_return = np.zeros(num_agents)
        done = False

        while not done:
            if config.get("video") and i == eval_episodes - 1:
                video.record_frame(env)
            actions = eval_agents.act(obss)
            obss, rewards, done, _, _ = env.step(actions)
            episodic_return += rewards

        if config.get("video") and i == eval_episodes - 1:
            video.record_frame(env)
            video.save(f"{config['dir']}/video/eval-rep-{rep_num}.mp4")
            video.reset()

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns, axis=0)
    std_return = np.std(episodic_returns, axis=0)

    if config.get("output", True):
        print(f"  Rep {rep_num + 1} EVALUATION:")
        print(f"    Agent 1 ({config['algorithm_1']}): {mean_return[0]:.2f} ± {std_return[0]:.2f}")
        print(f"    Agent 2 ({config['algorithm_2']}): {mean_return[1]:.2f} ± {std_return[1]:.2f}")

    return episodic_returns


def run_multiple_repetitions(env, config):
    """
    Run multiple independent training repetitions and aggregate results.

    :param env (gym.Env): environment to use
    :param config (Dict): configuration dictionary
    :return: aggregated evaluation results and final Q-tables from all repetitions
    """
    all_eval_returns = []
    all_q_tables = []

    for rep in range(config["repetitions"]):
        print(f"\n{'='*50}")
        print(f"REPETITION {rep + 1}/{config['repetitions']}")
        print(f"{'='*50}")

        # Set seed for this repetition if base seed is provided
        if config["seed"] is not None:
            rep_seed = config["seed"] + rep
            random.seed(rep_seed)
            np.random.seed(rep_seed)

        # Train agents
        trained_agents = train_agents(env, config, rep)
        
        # Evaluate only at the end
        eval_returns = evaluate_agents(env, config, trained_agents, rep)
        
        all_eval_returns.append(eval_returns)
        all_q_tables.append(copy.deepcopy(trained_agents.q_tables))

    return all_eval_returns, all_q_tables


def save_results(all_eval_returns, config):
    """
    Save aggregated results to CSV with one row per repetition.

    :param all_eval_returns: list of evaluation returns per repetition
    :param config: configuration dictionary
    """
    import csv
    
    save_path = f"{config['dir']}/eval_returns.csv"
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['repetition', 'agent_1_returns', 'agent_2_returns'])
        
        for rep_idx, rep_returns in enumerate(all_eval_returns):
            agent_1_returns = [returns[0] for returns in rep_returns]
            agent_2_returns = [returns[1] for returns in rep_returns]
            writer.writerow([rep_idx + 1, agent_1_returns, agent_2_returns])
    
    print(f"\nResults saved to {save_path}")


def print_summary(all_eval_returns, config):
    """
    Print summary statistics across all repetitions.

    :param all_eval_returns: list of evaluation returns per repetition
    :param config: configuration dictionary
    """
    # Flatten all returns: shape (repetitions, eval_episodes, num_agents)
    all_returns = np.array(all_eval_returns)
    
    # Mean per repetition, then mean across repetitions
    rep_means = np.mean(all_returns, axis=1)  # (repetitions, num_agents)
    
    overall_mean = np.mean(rep_means, axis=0)
    overall_std = np.std(rep_means, axis=0)
    
    print(f"\n{'='*50}")
    print("SUMMARY ACROSS ALL REPETITIONS")
    print(f"{'='*50}")
    print(f"Agent 1 ({config['algorithm_1']}): {overall_mean[0]:.2f} ± {overall_std[0]:.2f}")
    print(f"Agent 2 ({config['algorithm_2']}): {overall_mean[1]:.2f} ± {overall_std[1]:.2f}")
    print(f"Total repetitions: {config['repetitions']}")
    print(f"Eval episodes per rep: {config['eval_episodes']}")


if __name__ == "__main__":
    # Setup save directory
    if CONFIG["save"]:
        CONFIG["runname"] = (
            f"{CONFIG['repetitions']}reps_{CONFIG['total_eps']}eps_{CONFIG['ep_length']}epL_"
            f"{CONFIG['runname']}_{CONFIG['algorithm_1']}_vs_{CONFIG['algorithm_2']}"
        )
        save_dir = f"{dirpath}output/MixedPlay/{CONFIG['runname']}"
        CONFIG["dir"] = save_dir

        if CONFIG["env"][-1] == "f" and CONFIG["visualise"] == True:
            CONFIG["video"] = True
            os.makedirs(f"{save_dir}/video", exist_ok=True)
        else:
            CONFIG["video"] = False
            os.makedirs(f"{save_dir}", exist_ok=True)

    # Set seeds
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # Environments
    if CONFIG["env"] == "cf":
        env = CustomForagingEnv(
            field_size=(5, 5),  
            players=len(CONFIG["player_pos"]),       
            min_player_level=2,
            max_player_level=3,
            min_food_level=1,
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
        env = gym.make(
            "lbforaging:Foraging-5x5-2p-1f-v3", 
            render_mode="rgb_array" if CONFIG.get("visualise") else None
        )

    elif CONFIG["env"] == "m":
        env = create_matrix_game(CONFIG["payoff_matrix"], CONFIG["ep_length"])
        CONFIG["video"] = False
        CONFIG["gamma"] = 0.0  # No bootstrapping for stateless matrix games
    
    elif CONFIG["env"] == "mc":
        env = MoveChairGame(ep_length=CONFIG["ep_length"])
        CONFIG["video"] = False

    else:
        raise ValueError(f"Invalid env '{CONFIG['env']}'. Choose 'f', 'cf', 'm', or 'mc'.")
    
    print(f"Starting Multiple Repetitions: {CONFIG['algorithm_1']} vs {CONFIG['algorithm_2']}")
    print(f"Environment: {CONFIG['env']}")
    print(f"Repetitions: {CONFIG['repetitions']}")
    print(f"Episodes per repetition: {CONFIG['total_eps']}")
    print("-" * 50)

    # Run multiple repetitions
    all_eval_returns, all_q_tables = run_multiple_repetitions(env, CONFIG)

    # Save and summarize results
    if CONFIG["save"]:
        save_results(all_eval_returns, CONFIG)
    
    print_summary(all_eval_returns, CONFIG)

    # Visualize final Q-tables from last repetition
    if CONFIG.get("visualise"):
        visualise_q_tables(all_q_tables[-1], save_path=CONFIG.get("dir") if CONFIG.get("save") else None)

