import random
import os
import copy
from datetime import datetime

import gymnasium as gym
import numpy as np

from utils.visualizations import (
    visualise_q_tables,
    visualise_evaluation_returns,
    visualise_repetition_returns,
    visualise_learning_curve,
)
from utils.video import VideoRecorder

from agent.mixed_play_wrapper import MixedPlay
from agent.iql import IQL
from agent.iql_unc import IQLAE
from agent.random_agent import Random
from agent.jal import JalAM
from agent.jal_unc import JalAE
from agent.iql_behave_managing import QBM
from agent.p_random import pRandom

from envs.matrix_game import create_matrix_game
from envs.custom_foraging_env import CustomForagingEnv
from envs.custom_foraging_oneFood import CustomForagingOneFood
from envs.move_game import MoveChairEnv
from envs.move_chair_simple import MoveChairSimple
from envs.move_game_coor import MoveChairCoordination

dirpath = "tabular_marl/"

# Available algorithms for mixed play
ALGORITHMS = {
    "Random": Random,
    "pRandom": pRandom,
    "IQL": IQL,
    "IQLAE": IQLAE,
    "JalAM": JalAM,
    "JalAE": JalAE,
    "QBM": QBM
}

CONFIG = {
    "runname": datetime.now().strftime("%d%b%Y").lower(),  # e.g. "15dec2025"
    
    # Mixed play configuration
    "algorithm_1": "JalAM",   # Algorithm for agent 1
    "algorithm_2": "JalAM",   # Algorithm for agent 2
    "algorithm_1_kwargs": {"lr": 0.2},  # pRandom here! extra kwargs for algorithm 1
    "algorithm_2_kwargs": {"lr": 0.2},  # Extra kwargs for algorithm 2

    "env": "mc",  # game type: "f" = foraging, "cf" = custom_foraging, "cf1f" = OneFood, "m" = matrix, "mc" = MoveChairGame

    "save": True,
    "visualise": False,
    "output": True,

    "repetitions": 2,  # Number of independent runs
    "ep_length": 50, 
    "total_eps": 3000,
    "eval_episodes": 100, #in total across one rep.
    "eval_spread": "both",  # "last10", "full", or "both" (saves 2 CSVs, uses last10 for repetition plot, full for learning curve)

    "seed": None,
    "lr": 0.1, # default 0.1 (for QBM: 0.25
    "init_epsilon": 0.9, #default 0.9
    "eps_decay": True, # False sets epsilon to 0.1
    "eval_epsilon": 0.05,
    "num_agents": 2,
    "gamma": 0.95, #default 0.95

    "food_pos": [[1, 1], [3, 3]],
    "player_pos": [[0, 4], [4, 0]],
    "payoff_matrix": np.array([[[5, 5], [0, 3]], 
                               [[3, 0], [2, 2]]]) * 1/10
}


def train_agents(env, config, rep_num=0):
    """
    Train agents with evaluation during the last 10% of training.

    :param env (gym.Env): environment to train on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param rep_num (int): repetition number for logging
    :return: trained agents and evaluation results from last 10% checkpoints
    """
    # Create mixed play agents
    agents = MixedPlay(
        num_agents=len(config["player_pos"]),
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        init_epsilon=config["init_epsilon"],
        eps_decay=config["eps_decay"],
        env=config["env"],
        algorithm_1=ALGORITHMS[config["algorithm_1"]],
        algorithm_2=ALGORITHMS[config["algorithm_2"]],
        algorithm_1_kwargs=config.get("algorithm_1_kwargs", {}),
        algorithm_2_kwargs=config.get("algorithm_2_kwargs", {}),
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["ep_length"]
    total_eps = config["total_eps"]
    
    # # Compute both checkpoint sets (used for "both" mode and individual modes)
    # full_checkpoints = set(int(total_eps * pct / 100) for pct in range(10, 101, 10))
    # full_checkpoints.discard(0)
    # eval_start = int(0.9 * total_eps)
    # last10_checkpoints = set(range(eval_start + 1, total_eps + 1))
    
    # # Determine which checkpoints to evaluate at
    # eval_spread = config.get("eval_spread", "last10")
    
    # if eval_spread == "both":
    #     eval_checkpoints = full_checkpoints | last10_checkpoints
    #     num_checkpoints = len(eval_checkpoints) # 10*2 = 20
    # elif eval_spread == "full":
    #     eval_checkpoints = full_checkpoints
    #     num_checkpoints = len(eval_checkpoints)
    # else:  # "last10" (default)
    #     eval_checkpoints = last10_checkpoints
    #     num_checkpoints = len(eval_checkpoints)

    # NEW
    # Compute both checkpoint sets
    eval_eps = config.get("eval_episodes", 100)
    full_checkpoints = set(int(total_eps * pct / 100) for pct in range(10, 101, 10))
    full_checkpoints.discard(0)
    eval_start = int(0.9 * total_eps)
    last10_checkpoints = set(range(eval_start + 1, total_eps + 1, total_eps // eval_eps))
    
    # Determine which checkpoints to evaluate at
    eval_spread = config.get("eval_spread", "last10")
    
    if eval_spread == "both":
        eval_checkpoints = full_checkpoints | last10_checkpoints
        # FIX: Divide budget by len(full_checkpoints) (10) instead of len(eval_checkpoints) (39)
        # This ensures the Learning Curve gets the full 100 episodes (10 per point)
        # The Last10 will also get 10 per point (higher quality, more time consuming)
        num_checkpoints = len(full_checkpoints)
        print(num_checkpoints)
    elif eval_spread == "full":
        eval_checkpoints = full_checkpoints
        num_checkpoints = len(eval_checkpoints)
    else:  # "last10" (default)
        eval_checkpoints = last10_checkpoints
        num_checkpoints = len(eval_checkpoints)
    
    # Evaluation episodes per checkpoint
    eval_eps_per_checkpoint = max(1, config["eval_episodes"] // num_checkpoints)

    # NEW ENDS
    
    # Evaluation episodes per checkpoint
    eval_eps_per_checkpoint = max(1, config["eval_episodes"] // num_checkpoints)
    
    checkpoint_returns = []  # Store returns from each checkpoint
    
    full_returns = [] if eval_spread == "both" else None
    last10_returns = [] if eval_spread == "both" else None

    for eps_num in range(1, 1 + total_eps):
        obss, _ = env.reset()
        done = False

        while not done:
            agents.schedule_hyperparameters(step_counter, max_steps)
            acts = agents.act(obss)
            n_obss, rewards, done, _, _ = env.step(acts)
            agents.learn(obss, acts, rewards, n_obss, done)

            step_counter += 1
            obss = n_obss

        # Evaluate at checkpoints
        if eps_num in eval_checkpoints:
            checkpoint_eval = evaluate_agents(
                env, config, agents, rep_num, 
                eval_episodes=eval_eps_per_checkpoint,
                checkpoint_pct=int(100 * eps_num / total_eps),
                verbose=False
            )
            
            if eval_spread == "both":
                if eps_num in full_checkpoints:
                    full_returns.extend(checkpoint_eval)
                if eps_num in last10_checkpoints:
                    last10_returns.extend(checkpoint_eval)
            else:
                checkpoint_returns.extend(checkpoint_eval)
            
            if config.get("output"):
                pct = int(100 * eps_num / total_eps)
                print(f"  Rep {rep_num + 1}: Episode {eps_num}/{total_eps} ({pct}%) - Evaluated")

    if eval_spread == "both":
        return agents, {"full": full_returns, "last10": last10_returns}
    return agents, checkpoint_returns


def evaluate_agents(env, config, trained_agents, rep_num=0, eval_episodes=None, checkpoint_pct=None, verbose=True):
    """
    Evaluate trained agents.

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict): configuration dictionary
    :param trained_agents (MixedPlay): the trained mixed play agents
    :param rep_num (int): repetition number for video naming
    :param eval_episodes (int): number of evaluation episodes (overrides config if provided)
    :param checkpoint_pct (int): training percentage at this checkpoint (for logging)
    :param verbose (bool): whether to print detailed evaluation results
    :return (list): list of episodic returns per evaluation episode
    """
    num_agents = len(config["player_pos"])
    eval_episodes = eval_episodes if eval_episodes is not None else config["eval_episodes"]

    video = VideoRecorder()
    
    # Create evaluation agents with low epsilon (greedy)
    eval_agents = MixedPlay(
        num_agents=num_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        init_epsilon=config["eval_epsilon"],
        eps_decay=config["eps_decay"],
        env=config["env"],
        algorithm_1=ALGORITHMS[config["algorithm_1"]],
        algorithm_2=ALGORITHMS[config["algorithm_2"]],
        algorithm_1_kwargs=config.get("algorithm_1_kwargs", {}),
        algorithm_2_kwargs=config.get("algorithm_2_kwargs", {}),
    )
    
    # Copy trained model (Q-tables + opponent model for JAL) to evaluation agents
    eval_agents.copy_model_from(trained_agents)

    episodic_returns = []
    for i in range(eval_episodes):
        obss, _ = env.reset()
        episodic_return = np.zeros(num_agents)
        done = False

        while not done:
            if config.get("video") and i == eval_episodes - 1 and checkpoint_pct == 100:
                video.record_frame(env)
            actions = eval_agents.act(obss)
            obss, rewards, done, _, _ = env.step(actions)
            episodic_return += rewards

        if config.get("video") and i == eval_episodes - 1 and checkpoint_pct == 100:
            video.record_frame(env)
            video.save(f"{config['dir']}/video/eval-rep-{rep_num}.mp4")
            video.reset()

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns, axis=0)
    std_return = np.std(episodic_returns, axis=0)

    if verbose and config.get("output", True):
        checkpoint_str = f" (at {checkpoint_pct}%)" if checkpoint_pct else ""
        print(f"  Rep {rep_num + 1} EVALUATION{checkpoint_str}:")
        print(f"    Agent 1 ({config['algorithm_1']}): {mean_return[0]:.2f} ± {std_return[0]:.2f}")
        print(f"    Agent 2 ({config['algorithm_2']}): {mean_return[1]:.2f} ± {std_return[1]:.2f}")

    return episodic_returns


def run_multiple_repetitions(env, config):
    """
    Run multiple independent training repetitions and aggregate results.
    Evaluates during the last 10% of training at 10 checkpoints.

    :param env (gym.Env): environment to use
    :param config (Dict): configuration dictionary
    :return: aggregated evaluation results and final Q-tables from all repetitions
             When eval_spread=="both", returns dict with "full" and "last10" keys
    """
    eval_spread = config.get("eval_spread", "last10")
    
    if eval_spread == "both":
        all_eval_returns = {"full": [], "last10": []}
    else:
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

        trained_agents, checkpoint_returns = train_agents(env, config, rep)
        
        if eval_spread == "both":
            all_eval_returns["full"].append(checkpoint_returns["full"])
            all_eval_returns["last10"].append(checkpoint_returns["last10"])
        else:
            all_eval_returns.append(checkpoint_returns)
        all_q_tables.append(copy.deepcopy(trained_agents.q_tables))

    return all_eval_returns, all_q_tables


def save_results(all_eval_returns, config):
    """
    Save aggregated results to CSV with one row per repetition.
    When eval_spread=="both", saves two separate files.

    :param all_eval_returns: list of evaluation returns per repetition, or dict with "full"/"last10" keys
    :param config: configuration dictionary
    """
    import csv
    
    def _save_csv(returns_list, filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['repetition', 'agent_1_returns', 'agent_2_returns'])
            for rep_idx, rep_returns in enumerate(returns_list):
                agent_1_returns = [returns[0] for returns in rep_returns]
                agent_2_returns = [returns[1] for returns in rep_returns]
                writer.writerow([rep_idx + 1, agent_1_returns, agent_2_returns])
        print(f"Results saved to {filepath}")
    
    if config.get("eval_spread") == "both":
        _save_csv(all_eval_returns["full"], f"{config['dir']}/eval_returns_full.csv")
        _save_csv(all_eval_returns["last10"], f"{config['dir']}/eval_returns_last10.csv")
    else:
        _save_csv(all_eval_returns, f"{config['dir']}/eval_returns.csv")


def print_summary(all_eval_returns, config):
    """
    Print summary statistics across all repetitions.

    :param all_eval_returns: list of evaluation returns per repetition, or dict with "full"/"last10" keys
    :param config: configuration dictionary
    """
    eval_spread = config.get("eval_spread", "last10")
    
    # Use last10 returns for summary when "both" (final performance)
    returns_for_summary = all_eval_returns["last10"] if eval_spread == "both" else all_eval_returns
    
    # Flatten all returns: shape (repetitions, eval_episodes_total, num_agents)
    all_returns = np.array(returns_for_summary)
    
    # Mean per repetition, then mean across repetitions
    rep_means = np.mean(all_returns, axis=1)  # (repetitions, num_agents)
    
    overall_mean = np.mean(rep_means, axis=0)
    overall_std = np.std(rep_means, axis=0)
    
    # Calculate checkpoint info based on eval_spread
    total_eps = config["total_eps"]
    
    if eval_spread == "both":
        checkpoint_desc = "both: 10 (full) + last10%"
    elif eval_spread == "full":
        checkpoint_desc = "10 (evenly spaced at 10-100%)"
    else:
        eval_start = int(0.9 * total_eps)
        num_checkpoints = total_eps - eval_start
        checkpoint_desc = f"{num_checkpoints} (at 91-100% of training)"
    
    print(f"\n{'='*50}")
    print("SUMMARY ACROSS ALL REPETITIONS")
    print(f"{'='*50}")
    print(f"Agent 1 ({config['algorithm_1']}): {overall_mean[0]:.2f} ± {overall_std[0]:.2f}")
    print(f"Agent 2 ({config['algorithm_2']}): {overall_mean[1]:.2f} ± {overall_std[1]:.2f}")
    print(f"Total repetitions: {len(returns_for_summary)}")
    print(f"Eval checkpoints: {checkpoint_desc}")


if __name__ == "__main__":
    # Setup save directory
    if CONFIG["save"]:
        CONFIG["runname"] = (
            f"{CONFIG['algorithm_1']}_vs_{CONFIG['algorithm_2']}_env-{CONFIG['env']}_{CONFIG['repetitions']}reps_{CONFIG['total_eps']}eps_{CONFIG['ep_length']}epL_"
            f"{CONFIG['runname']}"
        )
        if CONFIG.get("repetitions") == 30:
            save_dir = f"{dirpath}output/Final/{CONFIG['runname']}"
        else:
            save_dir = f"{dirpath}output/Multiple/{CONFIG['runname']}"

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
            min_player_level=3,
            max_player_level=3,
            min_food_level=1,
            max_food_level=5,
            max_num_food=len(CONFIG["food_pos"]),  
            sight=5,         
            max_episode_steps=CONFIG["ep_length"],  
            force_coop=False,
            pos_foods=CONFIG["food_pos"],
            pos_players=CONFIG["player_pos"],
            render_mode="rgb_array" if CONFIG.get("visualise") else None,
        )   
        env.reset()
    elif CONFIG["env"] == "cf1f":
        env = CustomForagingOneFood(
            field_size=(5, 5),  
            players=len(CONFIG["player_pos"]),       
            min_player_level=3,
            max_player_level=3,
            min_food_level=1,
            max_food_level=5,
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
        env = MoveChairEnv(ep_length=CONFIG["ep_length"])
        CONFIG["video"] = False
    
    elif CONFIG["env"] == "mcs":
        env = MoveChairSimple(ep_length=CONFIG["ep_length"])
        CONFIG["video"] = False
    
    elif CONFIG["env"] == "mcc":
        env = MoveChairCoordination(ep_length=CONFIG["ep_length"])
        CONFIG["video"] = False
    else:
        raise ValueError(f"Invalid env '{CONFIG['env']}'. Choose 'f', 'cf', 'm', or 'mc'.")
    
    eval_spread = CONFIG.get("eval_spread", "last10")
    if eval_spread == "both":
        checkpoint_desc = "both: 10 at 10-100% (full) + last10%"
        num_checkpoints = 10
    elif eval_spread == "full":
        num_checkpoints = 10
        checkpoint_desc = "10 checkpoints at 10-100%"
    else:
        num_checkpoints = CONFIG["total_eps"] - int(0.9 * CONFIG["total_eps"])
        checkpoint_desc = f"{num_checkpoints} checkpoints at 91-100%"
    eval_eps_per_checkpoint = max(1, CONFIG["eval_episodes"] // num_checkpoints)
    
    print(f"Starting Multiple Repetitions: {CONFIG['algorithm_1']} vs {CONFIG['algorithm_2']}")
    print(f"Environment: {CONFIG['env']}")
    print(f"Repetitions: {CONFIG['repetitions']}")
    print(f"Episodes per repetition: {CONFIG['total_eps']}")
    print(f"Evaluation: {checkpoint_desc} ({eval_eps_per_checkpoint} eps each)")
    print("-" * 50)

    # Run multiple repetitions
    all_eval_returns, all_q_tables = run_multiple_repetitions(env, CONFIG)

    # Save and summarize results
    if CONFIG["save"]:
        save_results(all_eval_returns, CONFIG)
    
    print_summary(all_eval_returns, CONFIG)
    
    # Visualize evaluation returns across repetitions (use last10 for "both")
    rep_returns_data = all_eval_returns["last10"] if eval_spread == "both" else all_eval_returns
    config_for_rep = {**CONFIG, "eval_spread": "last10"} if eval_spread == "both" else CONFIG
    fig_rep = visualise_repetition_returns(rep_returns_data, config_for_rep)
    if CONFIG["save"]:
        fig_rep.savefig(f"{CONFIG['dir']}/{CONFIG['runname']}_repetition_returns.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {CONFIG['dir']}/{CONFIG['runname']}_repetition_returns.png")
    
    # Visualize learning curve (use full for "both")
    lc_returns_data = all_eval_returns["full"] if eval_spread == "both" else all_eval_returns
    config_for_lc = {**CONFIG, "eval_spread": "full"} if eval_spread == "both" else CONFIG
    fig_lc = visualise_learning_curve(lc_returns_data, config_for_lc)
    if CONFIG["save"]:
        fig_lc.savefig(f"{CONFIG['dir']}/{CONFIG['runname']}_learning_curve.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {CONFIG['dir']}/{CONFIG['runname']}_learning_curve.png")

    # Visualize final Q-tables from last repetition
    if CONFIG.get("visualise"):
        visualise_q_tables(all_q_tables[-1], save_path=CONFIG.get("dir") if CONFIG.get("save") else None)

