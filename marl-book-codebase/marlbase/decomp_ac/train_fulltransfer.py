from collections import namedtuple
from pathlib import Path
import os
import torch
import numpy as np
from einops import rearrange
from gymnasium.spaces import flatdim
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import shutil

from marlbase.utils.video import VideoRecorder
from marlbase.utils.transfer import save_transferable_backbone, load_transferable_backbone
from marlbase.utils.register_rewards import RewardRegistry
from marlbase.utils.coord_score import CoordinationDetector

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
    """Main training routine for MAPPO-TCOT with Universal Basis Transfer."""
    cfg = DictConfig(cfg)
    device = cfg.model.device

    # 1. Resolve runtime output paths inside the Hydra run directory
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    orig_cwd = Path(hydra.utils.get_original_cwd())

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Overwrite targets for latest state
    registry_path = run_dir / "reward_registry.json"
    coord_detector_path = run_dir / "coord_detector.pt"

    # Intervals
    eval_interval = getattr(cfg, "eval_interval", 0) or 0
    save_interval = getattr(cfg, "save_interval", 0) or 0
    video_interval = getattr(cfg, "video_interval", 0) or 0

    # 2. Reward Registry & Pretrained Backbone Resolution
    registry = RewardRegistry(max_channels=getattr(cfg, "max_channels", 32))
    pretrained_backbone = cfg.get("pretrained_backbone", None)

    if pretrained_backbone is not None:
        pretrained_path = Path(pretrained_backbone)
        if not pretrained_path.is_absolute():
            pretrained_path = orig_cwd / pretrained_path

        source_registry_path = pretrained_path / "reward_registry.json"
        source_detector_path = pretrained_path / "coord_detector.pt"
        source_backbone_path = pretrained_path / "checkpoints" / "backbone_final.pt"
        
        if source_registry_path.exists():
            registry.load(str(source_registry_path))
            print(f"[Transfer] Paired with source registry: {source_registry_path}", flush=True)
        else:
            print(f"[Transfer] Warning: No registry found at {source_registry_path}. Fresh mapping.", flush=True)
    else:
        print("[Setup] Training from scratch: Initializing fresh RewardRegistry (slots 0..N-1).", flush=True)

    # 3. Channel Detection & Model Initialization
    env_unwrapped = eval_env.unwrapped
    channel_names = getattr(env_unwrapped, "channel_names", None) or getattr(
        env_unwrapped, "REWARD_CHANNELS", None
    )
    if channel_names is None:
        n_channels = getattr(env_unwrapped, "num_reward_channels", getattr(cfg, "n_reward_channels", 4))
        channel_names = [f"channel_{i}" for i in range(n_channels)]
        print(f"[Setup] Found no channel names! Created {len(channel_names)} semantic channels: {channel_names}", flush=True)
    else:
        print(f"[Setup] Detected {len(channel_names)} semantic channels: {channel_names}", flush=True)

    active_indices = registry.get_or_register(channel_names)
    active_indices_tensor = torch.tensor(active_indices, dtype=torch.long, device=device)
    registry.save(str(registry_path))

    model = hydra.utils.instantiate(
        cfg.model, 
        envs.single_observation_space, 
        envs.single_action_space, 
        cfg,
    )

    if pretrained_backbone is not None:
        load_transferable_backbone(model, str(source_backbone_path), device)

    # Initialize active channels on model and detector
    model.set_active_channels(active_indices_tensor, channel_names=channel_names)

    # Load transferred detector scores if available (mirrors registry.load)
    if pretrained_backbone is not None:
        if source_detector_path.exists():
            model.coord_detector.load(str(source_detector_path))
            print(f"[Transfer] Loaded CoordinationDetector from {source_detector_path}", flush=True)
            print(f"[Transfer] Seeded scores: {model.coord_detector.get_scores_dict()}", flush=True)
        else:
            print(f"[Transfer] Note: No detector found at {source_detector_path}. Fresh detector.", flush=True)

    logger.watch(model)

    # 4. Helper closures to eliminate boilerplate and ensure consistent states
    def save_checkpoint(tag: str):
        torch.save(model.state_dict(), str(checkpoint_dir / f"full_model_{tag}.pt"))
        save_transferable_backbone(model, str(checkpoint_dir / f"backbone_{tag}.pt"))
        registry.save(str(registry_path))
        model.coord_detector.save(str(coord_detector_path))

    def save_video(filename: str):
        model.eval()
        record_episodes(
            eval_env,
            model,
            cfg.video_frames,
            str(video_dir / filename),
            device,
        )
        model.train()

    # 5. Training Loop Setup
    parallel_envs = envs.observation_space[0].shape[0]
    step = 0
    updates = 0
    last_eval = 0
    last_save = 0
    last_video = 0
    accumulated_infos = []

    # Initial baseline video (Step 0)
    if video_interval > 0:
        save_video("step-0.mp4")

    # 6. Training Loop
    while step < cfg.total_steps:
        t, batch, rollout_infos = _collect_trajectories(
            envs,
            model,
            time_limit,
            parallel_envs,
            model.n_agents,
            device,
            cfg.use_proper_termination,
        )
        step += t * parallel_envs
        accumulated_infos.extend(rollout_infos)

        metrics = model.update(batch, step)
        accumulated_infos.append(metrics)
        updates += 1

        # Periodic evaluation logging
        if eval_interval > 0 and (step - last_eval) >= eval_interval:
            _log_progress(accumulated_infos, step, updates, logger)
            accumulated_infos = []
            last_eval = step

        # Periodic checkpointing
        if save_interval > 0 and (step - last_save) >= save_interval:
            save_checkpoint(f"s{step}")
            last_save = step

        # Periodic video recording
        if video_interval > 0 and (step - last_video) >= video_interval:
            save_video(f"step-{step}.mp4")
            last_video = step

    # 7. Final Checkpoint & Final Video
    save_checkpoint("final")

    if video_interval > 0:
        latest_video = video_dir / f"step-{last_video}.mp4"
        if last_video == step and latest_video.exists():
            shutil.copyfile(latest_video, video_dir / "step-final.mp4")
        else:
            save_video("step-final.mp4")

    # Clean Exit
    envs.close()
    if hasattr(eval_env, "close"):
        eval_env.close()