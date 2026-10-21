from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch

# Import record_episodes from your training script
from marlbase.decomp_optimappo_coorscore.train import record_episodes


def main(env, ckpt_path, **cfg):
    cfg = DictConfig(cfg)

    # 1. Instantiate the DecompOptiMAPPONetwork
    model = hydra.utils.instantiate(
        cfg.model, env.observation_space, env.action_space, cfg
    )

    # 2. Load trained checkpoint weights
    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Record video of the agents executing in LBF
    # Note: record_episodes only uses model.act() and env rendering;
    # decomposed rewards do not affect video recording.
    video_path = "./eval.mp4"
    record_episodes(
        env,
        model,
        cfg.video_frames,
        video_path,
        cfg.model.device,
    )
    print(f"Evaluation video saved successfully to {video_path}")

    env.close()