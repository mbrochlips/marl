# File: marlbase/decomp_ac/train.py

from collections import namedtuple
from pathlib import Path

from einops import rearrange
import numpy as np
from gymnasium.spaces import flatdim
import hydra
from omegaconf import DictConfig
import torch

from marlbase.utils.video import VideoRecorder
from marlbase.utils.transfer import save_transferable_backbone, load_transferable_backbone

# Batch structure supporting decomposed rewards and optional cached old log-probs
Batch = namedtuple(
    "Batch",
    ["obss", "actions", "rewards", "dones", "filled", "action_masks", "action_log_probs"],
    defaults=[None],  # action_log_probs is optional with default None
)


def _log_progress(infos, step, updates, logger):
    """Logs aggregated training and evaluation metrics."""
    infos.append({"updates": updates, "environment_steps": step})
    logger.log_metrics(infos)


def _collect_trajectories(
    envs, model, max_ep_length, parallel_envs, n_agents, device, use_proper_termination
):
    """
    Collects a trajectory rollout across parallel environments for Decomp-OptiMAPPO.
    
    Records:
      - Observations, actions, dones, and filled masks.
      - 4D Decomposed Rewards: (T, Parallel_Envs, N_Agents, N_Channels).
      - Old Action Log-Probabilities: cached for PPO ratio calculation.
      - Action masks (if supported by environment).
    """
    running = torch.ones(parallel_envs, device=device, dtype=torch.bool)

    # Reset environment and initialize observation tensors
    obss, info = envs.reset()
    obss = [torch.tensor(o, device=device) for o in obss]
    parallel_envs = envs.observation_space[0].shape[0]
    obs_dim = flatdim(envs.single_observation_space)
    num_actions = max(action_space.n for action_space in envs.single_action_space)

    # Query number of reward channels from the model
    n_reward_channels = getattr(model, "n_reward_channels", 3)

    # -------------------------------------------------------------
    # 1. Rollout Storage Pre-allocation
    # -------------------------------------------------------------
    batch_obs = torch.zeros(
        max_ep_length + 1,
        parallel_envs,
        obs_dim,
        device=device,
    )
    batch_done = torch.zeros(
        max_ep_length + 1, parallel_envs, device=device, dtype=torch.bool
    )
    batch_act = torch.zeros(
        max_ep_length, parallel_envs, n_agents, device=device, dtype=torch.long
    )
    # Shape: (T, Parallel_Envs, N_Agents, N_Channels)
    batch_rew = torch.zeros(
        max_ep_length, parallel_envs, n_agents, n_reward_channels, device=device
    )
    batch_filled = torch.zeros(max_ep_length, parallel_envs, device=device)

    # Storage for old action log probabilities across timesteps
    batch_log_probs = torch.zeros(
        max_ep_length, parallel_envs, n_agents, device=device
    )
    
    # Internal tracker for per-channel decomposed returns during rollout
    env_decomposed_returns = torch.zeros(
        parallel_envs, n_agents, n_reward_channels, device=device
    )

    t = 0
    infos = []

    # Action mask initialization
    if "action_mask" in info:
        batch_action_masks = torch.ones(
            max_ep_length + 1, parallel_envs, n_agents, num_actions, device=device
        )
        mask = np.stack(info["action_mask"], dtype=np.float32)
        action_mask = torch.tensor(mask, dtype=torch.float32, device=device)
        batch_action_masks[0] = action_mask
        action_mask = action_mask.swapaxes(0, 1)
    else:
        batch_action_masks = None
        action_mask = None

    batch_obs[0] = torch.cat(obss, dim=-1)
    actor_hiddens = model.init_actor_hiddens(parallel_envs)

    # -------------------------------------------------------------
    # 2. Rollout Collection Loop
    # -------------------------------------------------------------
    while running.any() and t < max_ep_length:
        with torch.no_grad():
            # Query policy actions and evaluate log probabilities
            act_output = model.act(
                obss,
                actor_hiddens,
                action_mask=action_mask,
            )

            # Robust handling if model.act returns (actions, hiddens) or (actions, hiddens, log_probs)
            if len(act_output) == 3:
                actions, actor_hiddens, log_probs = act_output
            else:
                actions, actor_hiddens = act_output
                # Compute log_probs using actor logits if not returned directly by act()
                inputs = [i.unsqueeze(0) for i in obss]
                logits, _ = model.actor(inputs, actor_hiddens)
                logits = [l.squeeze(0) for l in logits]
                dist = model.get_dist(logits, action_mask)
                log_probs = torch.cat(dist.log_probs(actions), dim=-1)  # (parallel_envs, n_agents)

        # Step the parallel environments
        next_obss, rewards, done, truncated, info = envs.step(
            actions.squeeze().tolist()
        )
        next_obss = [torch.tensor(o, device=device) for o in next_obss]

        done = torch.tensor(done, dtype=torch.bool, device=device)
        truncated = torch.tensor(truncated, dtype=torch.bool, device=device)
        if not use_proper_termination:
            done = torch.logical_or(done, truncated)

        # Convert vector rewards cleanly to tensor: (parallel_envs, n_agents, n_channels)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        if rewards_tensor.ndim == 2:
            rewards_tensor = rewards_tensor.unsqueeze(0)

        # Record step transitions
        batch_obs[t + 1, running, :] = torch.cat(next_obss, dim=1)[running]
        batch_act[t, running] = rearrange(actions, "N B 1 -> B N")[running]
        batch_done[t + 1, running] = done[running]
        batch_rew[t, running] = rewards_tensor[running]
        batch_filled[t, running] = 1.0
        batch_log_probs[t, running] = log_probs[running]

        # Accumulate per-channel episodic return
        env_decomposed_returns[running] += rewards_tensor[running]

        if "action_mask" in info:
            mask = np.stack(info["action_mask"], dtype=np.float32)
            action_mask = torch.tensor(mask, dtype=torch.float32, device=device)
            batch_action_masks[t + 1, running] = action_mask[running]
            action_mask = action_mask.swapaxes(0, 1)

        # -------------------------------------------------------------
        # 3. Episode Completion & Decomposed Metric Logging
        # -------------------------------------------------------------
        if done.any():
            for i, d in enumerate(done):
                if d and running[i]:
                    ep_info = {}

                    # Copy numeric scalar metadata from final_info
                    if "final_info" in info and info["final_info"][i] is not None:
                        for k, v in info["final_info"][i].items():
                            if isinstance(v, (int, float, np.integer, np.floating)):
                                ep_info[k] = float(v)

                    # Calculate decomposed episodic returns
                    agent_ch_returns = env_decomposed_returns[i].detach().cpu().numpy()  # (n_agents, n_channels)
                    total_scalar_return = float(agent_ch_returns.sum())
                    ep_info["episode_returns"] = total_scalar_return

                    # Dynamically log every active channel by name or index
                    n_channels = agent_ch_returns.shape[1]
                    channel_names = getattr(
                        envs.unwrapped,
                        "REWARD_CHANNELS",
                        [f"channel_{c}" for c in range(n_channels)],
                    )

                    for c in range(n_channels):
                        ch_name = channel_names[c] if c < len(channel_names) else f"channel_{c}"
                        ep_info[f"return_{ch_name}"] = float(agent_ch_returns[:, c].sum())

                    infos.append(ep_info)

                    # Reset tracking for this env index
                    env_decomposed_returns[i].zero_()
                    running[i] = False

        t += 1
        obss = next_obss

    batch = Batch(
        batch_obs,
        batch_act,
        batch_rew,
        batch_done,
        batch_filled,
        batch_action_masks,
        batch_log_probs,
    )

    return t, batch, infos


def record_episodes(env, model, n_timesteps, path, device):
    """Renders evaluation episodes and writes out a video file."""
    recorder = VideoRecorder()
    done = True

    for _ in range(n_timesteps):
        if done:
            obss, info = env.reset()
            hiddens = model.init_actor_hiddens(1)
            if "action_mask" in info:
                action_mask = torch.tensor(
                    info["action_mask"], dtype=torch.float32, device=device
                )
            else:
                action_mask = None
            done = False
        else:
            with torch.no_grad():
                # obss = torch.tensor(obss, dtype=torch.float32, device=device).unsqueeze(1)
                obss = torch.as_tensor(np.array(obss), dtype=torch.float32, device=device).unsqueeze(1)
                act_output = model.act(obss, hiddens, action_mask)
                actions = act_output[0]
                hiddens = act_output[1]
                obss, _, done, truncated, info = env.step([a.item() for a in actions])
                if "action_mask" in info:
                    action_mask = torch.tensor(
                        info["action_mask"], dtype=torch.float32, device=device
                    )
            done = done or truncated
        recorder.record_frame(env)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    recorder.save(path)


def main(envs, eval_env, logger, time_limit, **cfg):
    cfg = DictConfig(cfg)

    # 1. Instantiate Model
    model = hydra.utils.instantiate(
        cfg.model, envs.single_observation_space, envs.single_action_space, cfg
    )

    # 2. Optional: Load Pretrained Transferable Backbone
    pretrained_backbone = getattr(cfg, "pretrained_backbone", None)
    if pretrained_backbone is not None and Path(pretrained_backbone).exists():
        load_transferable_backbone(model, pretrained_backbone, cfg.model.device)

    logger.watch(model)

    parallel_envs = envs.observation_space[0].shape[0]
    step = 0
    updates = 0
    last_eval = 0
    last_save = 0
    accumulated_infos = []

    while step < cfg.total_steps + 1:
        # Collect rollouts (exact same function as before)
        t, batch, rollout_infos = _collect_trajectories(
            envs,
            model,
            time_limit,
            parallel_envs,
            model.n_agents,
            cfg.model.device,
            cfg.use_proper_termination,
        )
        accumulated_infos.extend(rollout_infos)

        # Update model
        metrics = model.update(batch, step)
        accumulated_infos.append(metrics)

        # Logging interval
        if (step - last_eval) >= cfg.eval_interval:
            _log_progress(accumulated_infos, step, updates, logger)
            accumulated_infos = []
            last_eval = step

        # Checkpointing
        if cfg.save_interval and (step - last_save) >= cfg.save_interval:
            Path("checkpoints").mkdir(exist_ok=True)
            # Full checkpoint for resuming training on the SAME env
            torch.save(model.state_dict(), f"checkpoints/full_model_s{step}.pt")
            # Transferable backbone for transferring to ANY OTHER env
            save_transferable_backbone(model, f"checkpoints/backbone_s{step}.pt")
            last_save = step

        updates += 1
        step += t * parallel_envs

    # Save final transferable checkpoint
    save_transferable_backbone(model, "checkpoints/backbone_final.pt")
    envs.close()