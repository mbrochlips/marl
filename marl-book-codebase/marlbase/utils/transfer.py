import torch
import torch.nn as nn

EXCLUDED_PREFIXES = (
    "critic.channel_embeddings",
    "target_critic.channel_embeddings",
    "critic.film_ffn.film_proj",
    "target_critic.film_ffn.film_proj",
    "ret_ms",
    "coord_detector",
    "channel_weights",
)


def save_transferable_backbone(model: nn.Module, path: str):
    """
    Saves the agent's policy and critic backbone, stripping out environment-specific
    channel embeddings, FiLM projections, running statistics, and detector buffers.
    """
    state_dict = model.state_dict()
    transferable_dict = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in EXCLUDED_PREFIXES)
    }
    torch.save(transferable_dict, path)
    print(f"[Transfer] Saved {len(transferable_dict)} backbone tensors to {path}")


def load_transferable_backbone(model: nn.Module, path: str, device: torch.device):
    """
    Loads a pre-trained backbone into a newly initialized model.
    Guarantees:
      1. Strictly orthogonal channel query embeddings (zero overlap).
      2. Zero-initialized FiLM projections (identity pass-through at step 0).
      3. Clean target-critic synchronization.
    """
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = model.state_dict()

    # Filter out mismatched keys (if any)
    matched_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    model.load_state_dict(matched_dict, strict=False)

    # 1. Guarantee ZERO query overlap using exact orthogonal initialization
    nn.init.orthogonal_(model.critic.channel_embeddings.weight)

    # 2. Reset FiLM modulation to exact identity (gamma=0, beta=0)
    nn.init.zeros_(model.critic.film_ffn.film_proj.weight)
    nn.init.zeros_(model.critic.film_ffn.film_proj.bias)

    # 3. Synchronize target critic
    model.soft_update(1.0)

    print(f"[Transfer] Successfully loaded {len(matched_dict)} layers from {path}.")
    print(f"[Transfer] Re-initialized orthogonal queries (M={model.n_reward_channels}) and zero-init FiLM.")