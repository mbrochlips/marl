import random
import os
import copy
from datetime import datetime

import gymnasium as gym
import numpy as np

from utils.visualizations import (
    visualise_q_tables,
    visualise_q_convergence,
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
    "runname": datetime.now().strftime("%d%b%Y").lower(),  # e.g."15dec2025"
    
    # Mixed play configuration
    "algorithm_1": "IQL",   # Algorithm for agent 1
    "algorithm_2": "IQL",   # Algorithm for agent 2
    "algorithm_1_kwargs": {},  # Extra kwargs for algorithm 1
    "algorithm_2_kwargs": {}, #"p": 0.9},  # Extra kwargs for algorithm 2 (e.g., Random's p)
    
    "env": "mc",  # game type: "f" = foraging, "cf" = custom_foraging, "m" = matrix, "mc" = MoveChairGame

    "save": True,
    "visualise": False,
    "output": True,

    "ep_length": 100,
    "total_eps": 10000,
    "eval_freq": 1000,
    "eval_episodes": 100,

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


def train_mixed_agents(env, config):
    """
    Train and evaluate mixed play agents (two different algorithms) in env.

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return: evaluation results and Q-tables
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
    
    video = VideoRecorder()

    evaluation_return_means = []
    evaluation_return_stds = []
    evaluation_q_tables = []

    all_rewards = 0

    for eps_num in range(1, 1 + config["total_eps"]):
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

            if config.get("video"):
                video.record_frame(env)

        if eps_num > 0 and (eps_num) % config["eval_freq"] == 0:
            if config.get("video"):
                video.save(f"{config['dir']}/video/run-{eps_num}.mp4")
                video.reset()
                
            print(f"Episode {eps_num}/{config['total_eps']}")
            print(f"  {config['algorithm_1']} (Agent 1) epsilon: {agents.agent_1.epsilon:.4f}")
            print(f"  {config['algorithm_2']} (Agent 2) epsilon: {agents.agent_2.epsilon:.4f}")
            #print(f"  Q-tables agent_1 ({config['algorithm_1']}): {list(agents.q_tables[0].values())}")
            #print(f"  Q-tables agent_2 ({config['algorithm_2']}): {list(agents.q_tables[1].values())}")

            mean_return, std_return = evaluate_mixed(env, config, agents)
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            evaluation_q_tables.append(copy.deepcopy(agents.q_tables))
        else:
            if config.get("video"):
                video.reset()

    return (
        evaluation_return_means,
        evaluation_return_stds,
        evaluation_q_tables,
        agents.q_tables,
    )


def evaluate_mixed(env, config, trained_agents):
    """
    Evaluate mixed play agents using their trained Q-tables.

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict): configuration dictionary
    :param trained_agents (MixedPlay): the trained mixed play agents
    :return (ndarray, ndarray): mean and standard deviation of returns per agent
    """
    num_agents = len(config["player_pos"])
    eval_episodes = config["eval_episodes"]
    
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
    for _ in range(eval_episodes):
        obss, _ = env.reset()
        episodic_return = np.zeros(num_agents)
        done = False

        while not done:
            actions = eval_agents.act(obss)
            obss, rewards, done, _, _ = env.step(actions)
            episodic_return += rewards

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns, axis=0)
    std_return = np.std(episodic_returns, axis=0)

    if config.get("output", True):
        print("EVALUATION RETURNS:")
        print(f"  Agent 1 ({config['algorithm_1']}): {mean_return[0]:.2f} ± {std_return[0]:.2f}")
        print(f"  Agent 2 ({config['algorithm_2']}): {mean_return[1]:.2f} ± {std_return[1]:.2f}")
    
    print("--------------------------------")

    return mean_return, std_return


if __name__ == "__main__":
    # Setup save directory
    if CONFIG["save"]:
        CONFIG["runname"] = (
            f"{CONFIG['total_eps']}eps_{CONFIG['ep_length']}epL_"
            f"{CONFIG['runname']}_{CONFIG['algorithm_1']}_vs_{CONFIG['algorithm_2']}"
        )
        save_dir = f"{dirpath}output/MixedPlay/{CONFIG['runname']}"
        CONFIG["dir"] = save_dir

        if CONFIG["env"][-1] == "f":
            CONFIG["video"] = True
            os.makedirs(f"{save_dir}/video", exist_ok=True)
        else:
            CONFIG["video"] = False
            os.makedirs(f"{save_dir}", exist_ok=True)

    # Set seeds
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # Create environment
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
        raise ValueError(f"Invalid env '{CONFIG['env']}'. Choose 'f', 'cf', or 'm'.")
    
    print(f"Starting Mixed Play: {CONFIG['algorithm_1']} vs {CONFIG['algorithm_2']}")
    print(f"Environment: {CONFIG['env']}")
    print(f"Total episodes: {CONFIG['total_eps']}")
    print("-" * 50)

    # Train mixed play agents
    evaluation_return_means, evaluation_return_stds, eval_q_tables, final_q_tables = train_mixed_agents(
        env, CONFIG
    )

    # Visualize results
    fig = visualise_evaluation_returns(evaluation_return_means, evaluation_return_stds, CONFIG, dirpath)
    visualise_q_tables(final_q_tables, max_states=10, show_all_actions=True)
    print(f"\nFinal Q-table sizes: Agent 1: {len(final_q_tables[0])}, Agent 2: {len(final_q_tables[1])}")

