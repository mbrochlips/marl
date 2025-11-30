import copy
import random
#from gymnasium.envs.registration import register
import gymnasium as gym
import numpy as np

from video import VideoRecorder

from iql import IQL
from jal import JAL

from utils import (
    visualise_q_tables,
    visualise_q_convergence,
    visualise_evaluation_returns,
)
from matrix_game import create_pd_game
from custom_foraging_env import CustomForagingEnv


CONFIG = {
    "num_agents": 2,
    "algorithm": IQL,
    "seed": 1,
    "gamma": 0.99,
    "total_eps": 10,
    "ep_length": 50,
    "eval_freq": 50,
    "lr": 0.05,
    "init_epsilon": 0.9,
    "eval_epsilon": 0.05,
}

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
        episodic_return = np.zeros(config["num_agents"])
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
        num_agents= config["num_agents"],
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["init_epsilon"],
    )

    video = VideoRecorder()
    
    step_counter = 0
    max_steps = config["ep_length"] #config["total_eps"] * 

    evaluation_return_means = []
    evaluation_return_stds = []
    evaluation_q_tables = []

    for eps_num in range(config["total_eps"]):
        
        obss, _ = env.reset()
        episodic_return = np.zeros(CONFIG["num_agents"])
        done = False

        while not done:
            
            agents.schedule_hyperparameters(step_counter, max_steps)
            acts = agents.act(obss)
            n_obss, rewards, done, _, _ = env.step(acts)
            agents.learn(obss, acts, rewards, n_obss, done)
            
            step_counter += 1
            episodic_return += rewards
            obss = n_obss
            if eps_num > 0 and eps_num % config["eval_freq"] == 0:
                video.record_frame(env)

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, std_return = iql_eval(
                env, config, agents.q_tables, output=output
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            evaluation_q_tables.append(copy.deepcopy(agents.q_tables))
            video.save(f"/output/video_seed_{config['seed']}_eps_{eps_num}.mp4")
        

    return (
        evaluation_return_means,
        evaluation_return_stds,
        evaluation_q_tables,
        agents.q_tables,
    )


if __name__ == "__main__":
    #random.seed(CONFIG["seed"])
    #np.random.seed(CONFIG["seed"])

    stag_hunt = np.array([[[5, 5], [0, 4]], 
                              [[4, 0], [3, 3]]])
    
    #env = create_pd_game(stag_hunt,1)

    #env = gym.make("lbforaging:Foraging-5x5-2p-3f-v3", 25)
    pos_food = [[1,1],[3,3]]
    pos_player = [[0,5],[5,0]]

    env = CustomForagingEnv(
        field_size=(5, 5),          # Grid size
        players=2,                  # Two agents
        min_player_level=1,
        max_player_level=1,
        min_food_level=1,
        max_food_level=2,
        max_num_food=2,             # Two food items
        sight=5,                    # max
        max_episode_steps=CONFIG["ep_length"],       # Max steps per episode
        force_coop=False,
        pos_foods=pos_food,
        pos_players=pos_player,
        )   
    env.reset()
    
    evaluation_return_means, evaluation_return_stds, eval_q_tables, q_tables = train(
        env, CONFIG
    )
    
    #visualise_q_tables(q_tables)
    #visualise_evaluation_returns(evaluation_return_means, evaluation_return_stds)
    #visualise_q_convergence(eval_q_tables, env)