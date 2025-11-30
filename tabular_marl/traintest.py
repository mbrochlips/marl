import argparse
import logging
import time

import gymnasium as gym
import numpy as np

import lbforaging  # noqa

logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """ """
    obss, _ = env.reset()
    done = False

    returns = np.zeros(env.n_agents)

    if render:
        env.render()
        time.sleep(0.5)

    while not done:
        agents.schedule_hyperparameters(step_counter, max_steps)
        acts = agents.act(obss)
        n_obss, rewards, done, _, _ = env.step(acts)
        agents.learn(obss, acts, rewards, n_obss, done)
            
        step_counter += 1
        episodic_return += rewards
        obss = n_obss

        obss, rewards, done, _, _ = env.step(actions)


        if render:
            env.render()
            time.sleep(0.5)

    print("Returns: ", returns)


def main(episodes=1, render=False):
    env = gym.make("Foraging-8x8-2p-2f-v3")

    for episode in range(episodes):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--episodes", type=int, default=1, help="How many episodes to run"
    )

    args = parser.parse_args()
    main(args.episodes, args.render)