# File: marlbase/dummy/train.py
import os
import gymnasium as gym
import numpy as np
import pandas as pd


def main(*args, **kwargs):
    print("=" * 60)
    print("Starting Dummy Random Agent")
    print("=" * 60)

    # 1. Retrieve environment and logger passed by run.py
    env = kwargs.get("envs", None) or kwargs.get("eval_env", None) or kwargs.get("env", None)
    if env is None and len(args) > 0 and hasattr(args[0], "step"):
        env = args[0]

    # Fallback instantiation if run.py did not pass the env directly
    if env is None:
        env_name = kwargs.get("env_name", None)
        if env_name is None and len(args) > 0 and hasattr(args[0], "env"):
            env_name = args[0].env.name
        if env_name:
            env = gym.make(env_name)
        else:
            raise RuntimeError(f"Could not retrieve environment. Args={args}, Kwargs={kwargs.keys()}")

    logger = kwargs.get("logger", None)
    if logger is None and len(args) > 2:
        logger = args[2]

    num_episodes = kwargs.get("episodes", 5)
    channels = getattr(env.unwrapped, "REWARD_CHANNELS", ["ID_1", "ID_2", "ID_3"])
    num_agents = len(env.unwrapped.players)

    all_ep_returns = []
    all_ep_lengths = []
    total_env_steps = 0

    # 2. Rollout Loop (Random Actions)
    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_returns = np.zeros((num_agents, len(channels)), dtype=np.float32)
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            step_count += 1
            total_env_steps += 1

            actions = env.action_space.sample()
            step_res = env.step(actions)

            if len(step_res) == 5:
                obs, rewards, terminated, truncated, info = step_res
            else:
                obs, rewards, done, info = step_res
                terminated = done
                truncated = False

            # Accumulate vector rewards across agents
            for i in range(num_agents):
                ep_returns[i] += rewards[i]

        all_ep_returns.append(ep_returns)
        all_ep_lengths.append(step_count)

        print(f"[Episode {ep + 1}/{num_episodes}] Finished in {step_count} steps.")
        for i in range(num_agents):
            scalar_total = ep_returns[i].sum()
            breakdown = ", ".join([f"{ch}: {ep_returns[i, k]:.2f}" for k, ch in enumerate(channels)])
            print(f"  Agent {i} -> Total: {scalar_total:.2f} | Decomposed ({breakdown})")

    # 3. Compute Summary Metrics
    all_returns_arr = np.array(all_ep_returns)  # shape: (episodes, agents, channels)
    mean_total_return = float(all_returns_arr.sum(axis=-1).sum(axis=-1).mean())
    mean_ch0 = float(all_returns_arr[:, :, 0].sum(axis=-1).mean())
    mean_ch1 = float(all_returns_arr[:, :, 1].sum(axis=-1).mean())
    mean_ch2 = float(all_returns_arr[:, :, 2].sum(axis=-1).mean())
    mean_length = float(np.mean(all_ep_lengths))

    metrics = {
        "environment_steps": total_env_steps,
        "updates": num_episodes,
        "mean_episode_returns": mean_total_return,
        "mean_episode_length": mean_length,
        "return_ID_1_stoch": mean_ch0,
        "return_ID_2_solo": mean_ch1,
        "return_ID_3_coop": mean_ch2,
    }

    # 4. Log metrics via marl-base logger interface (list of dicts)
    infos = [metrics]

    if logger is not None:
        try:
            logger.log_metrics(infos)
        except Exception as e:
            print(f"Warning: logger.log_metrics failed: {e}")

        # Safety: Guarantee results.csv exists on disk so logger.get_state() never fails
        results_path = getattr(logger, "results_path", "results.csv")
        if not os.path.exists(results_path):
            df = pd.DataFrame([metrics])
            df.to_csv(results_path, index=True)
            print(f"Created fallback results file at: {results_path}")

    print("=" * 60)
    print("Dummy agent evaluation finished successfully!")
    print("=" * 60)