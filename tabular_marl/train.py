import numpy as np
import copy
from utils.video import VideoRecorder
from utils.eval import evaluate

from agent.random_agent import Random
from agent.iql import IQL
from agent.jal import JalAM

algorithms = {
    "IQL": IQL,
    "Random": Random,
}

def train_agents(env, config):
    """
    Train and evaluate independent Q-learning in env with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag if mean evaluation results should be printed
    :return (List[List[float]], List[List[float]], List[Dict[Act, float]]):
    """
    agents = algorithms[config["algorithm"]](
        num_agents=len(config["player_pos"]),
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        init_epsilon=config["init_epsilon"],
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
                
            print(f"episilon: {agents.epsilon}")
            print(f"q-tables agent_1: {agents.q_tables[0].values()}")
            print(f"q-tables agent_2: {agents.q_tables[1].values()}")

            mean_return, std_return = evaluate(
                env, config, agents.q_tables
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            evaluation_q_tables.append(copy.deepcopy(agents.q_tables))
        else:
            if config["video"]:
                video.reset()

    return (
        evaluation_return_means,
        evaluation_return_stds,
        evaluation_q_tables,
        agents.q_tables,  # all agents' Q-tables
    )