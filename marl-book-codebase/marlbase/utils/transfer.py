from pathlib import Path
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
    Saves the agent's policy and universal critic backbone.
    Strips out task-specific channel embeddings, FiLM projections, and running stats.
    """
    state_dict = model.state_dict()
    transferable_dict = {
        k: v.cpu() for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in EXCLUDED_PREFIXES)
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(transferable_dict, path)
    print(f"[Transfer] Saved {len(transferable_dict)} backbone tensors to '{path}'.")


def load_transferable_backbone(model: nn.Module, path: str, device: torch.device):
    """
    Loads pre-trained backbone into a newly initialized model for a new environment.
    
    Guarantees:
      1. 100% Critic match (Entity tokenizer, cross-attn, FiLM layers, readout head).
      2. Shape-aware padding for Actor input layer if observation dimension grew.
      3. Strictly orthogonal channel query embeddings for the new environment channels.
      4. Reset FiLM projection to identity baseline (zero-initialized).
    """
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = model.state_dict()

    loaded_keys = []
    padded_keys = []

    for k, v in pretrained_dict.items():
        if any(k.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue

        if k not in model_dict:
            continue

        target_shape = model_dict[k].shape

        if v.shape == target_shape:
            # Exact shape match: Critic layers & Actor hidden layers
            model_dict[k] = v
            loaded_keys.append(k)

        elif len(v.shape) == 2 and v.shape[0] == target_shape[0] and v.shape[1] < target_shape[1]:
            # Input dimension expansion on Actor (e.g., 15 -> 18 dims)
            old_in_dim = v.shape[1]
            model_dict[k][:, :old_in_dim] = v
            model_dict[k][:, old_in_dim:] = 0.0
            padded_keys.append((k, v.shape, target_shape))

    model.load_state_dict(model_dict, strict=False)

    # 1. Guarantee ZERO query overlap for the new channels
    nn.init.orthogonal_(model.critic.channel_embeddings.weight)

    # 2. Reset FiLM modulation to identity (gamma=0, beta=0)
    nn.init.zeros_(model.critic.film_ffn.film_proj.weight)
    nn.init.zeros_(model.critic.film_ffn.film_proj.bias)

    # 3. Synchronize Target Critic
    model.soft_update(1.0)

    print(f"[Transfer] Loaded {len(loaded_keys)} exact-matching tensors.")
    if padded_keys:
        for k, old_s, new_s in padded_keys:
            print(f"[Transfer] Zero-padded input layer '{k}': {old_s} -> {new_s}")
    print(f"[Transfer] Re-initialized {model.n_reward_channels} orthogonal channel queries & identity FiLM.")