from collections import namedtuple
from pathlib import Path
import os
import torch
import numpy as np
from einops import rearrange
from gymnasium.spaces import flatdim
import hydra
from omegaconf import DictConfig

from marlbase.utils.video import VideoRecorder
from marlbase.utils.transfer import save_transferable_backbone, load_transferable_backbone
from marlbase.utils.register_rewards import RewardRegistry

Batch = namedtuple(
    "Batch",
    ["obss", "actions", "rewards", "dones", "filled", "action_masks", "action_log_probs"],
    defaults=[None],
)


def _log_progress(infos, step, updates, logger):
    infos.append({"updates": updates, "environment_steps": step})
    logger.log_metrics(infos)


def _collect_trajectories(
    envs, model, max_ep_length, parallel_envs, n_agents, device, use_proper_termination
):
    running = torch.ones(parallel_envs, device=device, dtype=torch.bool)
    obss, info = envs.reset()
    obss = [torch.tensor(o, device=device) for o in obss]
    parallel_envs = envs.observation_space[0].shape[0]
    obs_dim = flatdim(envs.single_observation_space)
    num_actions = max(action_space.n for action_space in envs.single_action_space)

    n_reward_channels = model.n_reward_channels

    batch_obs = torch.zeros(max_ep_length + 1, parallel_envs, obs_dim, device=device)
    batch_done = torch.zeros(max_ep_length + 1, parallel_envs, device=device, dtype=torch.bool)
    batch_act = torch.zeros(max_ep_length, parallel_envs, n_agents, device=device, dtype=torch.long)
    batch_rew = torch.zeros(max_ep_length, parallel_envs, n_agents, n_reward_channels, device=device)
    batch_filled = torch.zeros(max_ep_length, parallel_envs, device=device)
    batch_log_probs = torch.zeros(max_ep_length, parallel_envs, n_agents, device=device)
    env_decomposed_returns = torch.zeros(parallel_envs, n_agents, n_reward_channels, device=device)

    t = 0
    infos = []

    if "action_mask" in info:
        batch_action_masks = torch.ones(max_ep_length + 1, parallel_envs, n_agents, num_actions, device=device)
        mask = np.stack(info["action_mask"], dtype=np.float32)
        action_mask = torch.tensor(mask, dtype=torch.float32, device=device)
        batch_action_masks[0] = action_mask
        action_mask = action_mask.swapaxes(0, 1)
    else:
        batch_action_masks, action_mask = None, None

    batch_obs[0] = torch.cat(obss, dim=-1)
    actor_hiddens = model.init_actor_hiddens(parallel_envs)

    while running.any() and t < max_ep_length:
        with torch.no_grad():
            act_output = model.act(obss, actor_hiddens, action_mask=action_mask)
            if len(act_output) == 3:
                actions, actor_hiddens, log_probs = act_output
            else:
                actions, actor_hiddens = act_output
                inputs = [i.unsqueeze(0) for i in obss]
                logits, _ = model.actor(inputs, actor_hiddens)
                logits = [l.squeeze(0) for l in logits]
                dist = model.get_dist(logits, action_mask)
                log_probs = torch.cat(dist.log_probs(actions), dim=-1)

        next_obss, rewards, done, truncated, info = envs.step(actions.squeeze().tolist())
        next_obss = [torch.tensor(o, device=device) for o in next_obss]

        done = torch.tensor(done, dtype=torch.bool, device=device)
        truncated = torch.tensor(truncated, dtype=torch.bool, device=device)
        if not use_proper_termination:
            done = torch.logical_or(done, truncated)

        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        if rewards_tensor.ndim == 2:
            rewards_tensor = rewards_tensor.unsqueeze(0)

        batch_obs[t + 1, running, :] = torch.cat(next_obss, dim=1)[running]
        batch_act[t, running] = rearrange(actions, "N B 1 -> B N")[running]
        batch_done[t + 1, running] = done[running]
        batch_rew[t, running] = rewards_tensor[running]
        batch_filled[t, running] = 1.0
        batch_log_probs[t, running] = log_probs[running]
        env_decomposed_returns[running] += rewards_tensor[running]

        if "action_mask" in info:
            mask = np.stack(info["action_mask"], dtype=np.float32)
            action_mask = torch.tensor(mask, dtype=torch.float32, device=device)
            batch_action_masks[t + 1, running] = action_mask[running]
            action_mask = action_mask.swapaxes(0, 1)

        if done.any():
            for i, d in enumerate(done):
                if d and running[i]:
                    ep_info = {}

                    # 1. Grab decomposed returns already computed by foraging_base.py
                    if "final_info" in info and info["final_info"][i] is not None:
                        for k, v in info["final_info"][i].items():
                            if isinstance(v, (int, float, np.integer, np.floating)):
                                ep_info[k] = float(v)

                    # 2. Total scalar return
                    agent_ch_returns = env_decomposed_returns[i].detach().cpu().numpy()
                    ep_info["episode_returns"] = float(agent_ch_returns.sum())

                    infos.append(ep_info)
                    env_decomposed_returns[i].zero_()
                    running[i] = False

        t += 1
        obss = next_obss

    batch = Batch(
        batch_obs, batch_act, batch_rew, batch_done, batch_filled, batch_action_masks, batch_log_probs
    )
    return t, batch, infos


def main(envs, eval_env, logger, time_limit, **cfg):
    """Main training routine for MAPPO-TCOT with Universal Basis Transfer."""
    cfg = DictConfig(cfg)
    device = cfg.model.device

    # -----------------------------------------------------------------
    # 1. Hydra Path Anchoring (Root Workspace Directory)
    # -----------------------------------------------------------------
    # Protects persistent assets from being trapped inside timestamped Hydra subfolders
    orig_cwd = (
        Path(hydra.utils.get_original_cwd())
        if hydra.core.hydra_config.HydraConfig.initialized()
        else Path(".")
    )
    checkpoint_dir = orig_cwd / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    registry_path = checkpoint_dir / "reward_registry.json"

    # -----------------------------------------------------------------
    # 2. Semantic Channel Discovery & Global Registry
    # -----------------------------------------------------------------
    registry = RewardRegistry(max_channels=getattr(cfg, "max_channels", 32))
    registry.load(str(registry_path))

    env_unwrapped = eval_env.unwrapped
    # Detect semantic names directly; avoid fallback to generic 'channel_0'
    channel_names = getattr(env_unwrapped, "channel_names", None) or getattr(
        env_unwrapped, "REWARD_CHANNELS", None
    )

    if channel_names is None:
        n_channels = getattr(
            env_unwrapped,
            "num_reward_channels",
            getattr(cfg, "n_reward_channels", 4),
        )
        channel_names = [f"channel_{i}" for i in range(n_channels)]

    print(f"[Setup] Detected {len(channel_names)} semantic channels: {channel_names}", flush=True)

    # Map semantic names to universal basis indices (e.g. [0, 1, 2, 8])
    active_indices = registry.get_or_register(channel_names)
    active_indices_tensor = torch.tensor(active_indices, dtype=torch.long, device=device)
    registry.save(str(registry_path))

    # -----------------------------------------------------------------
    # 3. Model Instantiation
    # -----------------------------------------------------------------
    model = hydra.utils.instantiate(
        cfg.model, envs.single_observation_space, envs.single_action_space, cfg
    )

    # -----------------------------------------------------------------
    # 4. Load Transferable Backbone (MUST happen before set_active_channels)
    # -----------------------------------------------------------------
    pretrained_backbone = cfg.get("pretrained_backbone", None)
    if pretrained_backbone is not None:
        pretrained_path = Path(pretrained_backbone)
        if not pretrained_path.is_absolute():
            pretrained_path = orig_cwd / pretrained_path

        if pretrained_path.exists():
            load_transferable_backbone(model, str(pretrained_path), device)
            print(f"[Transfer] Successfully loaded universal backbone from: {pretrained_path.resolve()}", flush=True)
        else:
            raise FileNotFoundError(f"[Error] Checkpoint not found at: {pretrained_path.resolve()}")

    # -----------------------------------------------------------------
    # 5. Bind Active Channels & Seed Coordination Detector
    # -----------------------------------------------------------------
    # Slices active basis queries, updates RunningMeanStd, and seeds coord_detector from loaded scores
    model.set_active_channels(active_indices_tensor)

    logger.watch(model)

    # -----------------------------------------------------------------
    # 6. Training Rollout & Update Loop
    # -----------------------------------------------------------------
    parallel_envs = envs.observation_space[0].shape[0]
    step = 0
    updates = 0
    last_eval = 0
    last_save = 0
    last_video = 0
    accumulated_infos = []

    while step < cfg.total_steps + 1:
        # Collect rollouts across parallel environments
        t, batch, rollout_infos = _collect_trajectories(
            envs,
            model,
            time_limit,
            parallel_envs,
            model.n_agents,
            device,
            cfg.use_proper_termination,
        )
        accumulated_infos.extend(rollout_infos)

        # Multi-epoch PPO mini-batch update
        metrics = model.update(batch, step)
        accumulated_infos.append(metrics)

        # Periodic Evaluation Logging
        if (step - last_eval) >= cfg.eval_interval:
            _log_progress(accumulated_infos, step, updates, logger)
            accumulated_infos = []
            last_eval = step

        # Periodic Checkpointing (Anchored to Root checkpoints/ folder)
        if cfg.save_interval and (step - last_save) >= cfg.save_interval:
            torch.save(model.state_dict(), str(checkpoint_dir / f"full_model_s{step}.pt"))
            save_transferable_backbone(model, str(checkpoint_dir / f"backbone_s{step}.pt"))
            registry.save(str(registry_path))
            last_save = step

    # -----------------------------------------------------------------
    # 7. Final Save & Clean Exit
    # -----------------------------------------------------------------
    torch.save(model.state_dict(), str(checkpoint_dir / "full_model_final.pt"))
    save_transferable_backbone(model, str(checkpoint_dir / "backbone_final.pt"))
    registry.save(str(registry_path))

    envs.close()
    if hasattr(eval_env, "close"):
        eval_env.close()