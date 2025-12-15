import numpy as np

from agent.random_agent import Random
from agent.iql import IQL
from agent.jal import JAL

algorithms = {
    "IQL": IQL,
    "JAL": JAL,
    "Random": Random,
}

def evaluate(env, config, q_tables):
    """
    Evaluate Q-learning agents on given environment using provided Q-tables.

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict): configuration dictionary containing hyperparameters
    :param q_tables (List[Dict]): Q-tables mapping states to action values for each agent
    :return (ndarray, ndarray): mean and standard deviation of returns per agent
    """
    num_agents = len(config["player_pos"])
    eval_episodes = config["eval_episodes"]
    
    eval_agents = algorithms[config["algorithm"]](
        num_agents=num_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        init_epsilon=config["eval_epsilon"],
    )
    eval_agents.q_tables = q_tables

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
        for i in range(num_agents):
            print(f"\tAgent {i+1}: {mean_return[i]:.2f} ± {std_return[i]:.2f}")
    
    return mean_return, std_return