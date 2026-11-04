# File: marlbase/utils/transfer.py

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
    "target_critic",  # Synced automatically via soft_update(1.0)
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
    print(f"[Transfer] Saved {len(transferable_dict)} backbone tensors to '{path}'.", flush=True)


def load_transferable_backbone(model: nn.Module, path: str, device: torch.device):
    """
    Robust Net2Net Transfer Loader for Option 1:
      1. Exact Match: Attention, FFN, Actor hidden layers.
      2. Input Zero-Padding: Actor & Critic inputs when food count (F) expands.
      3. Team-Residual Transfer: 
         - w_team and b_team carry over existing channel baselines.
         - delta_w and delta_b preserve existing agent offsets; new agents/channels stay at 0.0.
      4. Safe Re-initialization: Orthogonal channel queries & identity FiLM for new environment.
    """
    pretrained_dict = torch.load(path, map_location=device, weights_only=True)
    model_dict = model.state_dict()

    loaded_keys = []
    padded_keys = []
    residual_keys = []

    for k, v in pretrained_dict.items():
        if any(k.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue

        if k not in model_dict:
            continue

        target_shape = model_dict[k].shape

        # -------------------------------------------------------------
        # Case 1: Exact Shape Match
        # -------------------------------------------------------------
        if v.shape == target_shape:
            model_dict[k] = v
            loaded_keys.append(k)

        # -------------------------------------------------------------
        # Case 2: Input Dimension Expansion (e.g., Actor 15 -> 18, Critic 30 -> 36)
        # -------------------------------------------------------------
        elif len(v.shape) == 2 and v.shape[0] == target_shape[0] and v.shape[1] < target_shape[1]:
            old_in = v.shape[1]
            model_dict[k][:, :old_in] = v
            model_dict[k][:, old_in:] = 0.0
            padded_keys.append((k, v.shape, target_shape))

        # -------------------------------------------------------------
        # Case 3: Team Baseline Weights (w_team: M x d_model)
        # -------------------------------------------------------------
        elif "head.w_team" in k and len(v.shape) == 2 and v.shape[1] == target_shape[1] and v.shape[0] <= target_shape[0]:
            old_m = v.shape[0]
            model_dict[k][:old_m, :] = v
            residual_keys.append((k, f"Transferred {old_m} channel team weights"))

        # -------------------------------------------------------------
        # Case 4: Team Baseline Bias (b_team: M,)
        # -------------------------------------------------------------
        elif "head.b_team" in k and len(v.shape) == 1 and v.shape[0] <= target_shape[0]:
            old_m = v.shape[0]
            model_dict[k][:old_m] = v
            residual_keys.append((k, f"Transferred {old_m} channel team biases"))

        # -------------------------------------------------------------
        # Case 5: Agent Weight Offsets (delta_w: N x M x d_model) -> 3D Tensor!
        # -------------------------------------------------------------
        elif "head.delta_w" in k and len(v.shape) == 3 and v.shape[2] == target_shape[2]:
            old_n = min(v.shape[0], target_shape[0])
            old_m = min(v.shape[1], target_shape[1])
            # Slices existing agents and channels; new slots remain at 0.0
            model_dict[k][:old_n, :old_m, :] = v[:old_n, :old_m, :]
            residual_keys.append((k, f"Preserved {old_n} agents, {old_m} channels (new initialized to 0.0)"))

        # -------------------------------------------------------------
        # Case 6: Agent Bias Offsets (delta_b: N x M) -> 2D Tensor!
        # -------------------------------------------------------------
        elif "head.delta_b" in k and len(v.shape) == 2:
            old_n = min(v.shape[0], target_shape[0])
            old_m = min(v.shape[1], target_shape[1])
            model_dict[k][:old_n, :old_m] = v[:old_n, :old_m]
            residual_keys.append((k, f"Preserved {old_n} agent bias offsets"))

    # Load matched parameters into the model
    model.load_state_dict(model_dict, strict=False)

    # 1. Guarantee strictly orthogonal channel query embeddings for the active channels
    if hasattr(model.critic, "channel_embeddings"):
        nn.init.orthogonal_(model.critic.channel_embeddings.weight)

    # 2. Reset FiLM modulation to exact identity (gamma=0, beta=0)
    if hasattr(model.critic, "film_ffn"):
        nn.init.zeros_(model.critic.film_ffn.film_proj.weight)
        nn.init.zeros_(model.critic.film_ffn.film_proj.bias)

    # 3. Synchronize Target Critic
    model.soft_update(1.0)

    # Diagnostics printout
    print(f"[Transfer] Loaded {len(loaded_keys)} exact-matching tensors.", flush=True)
    if padded_keys:
        for k, old_s, new_s in padded_keys:
            print(f"[Transfer] Zero-padded input layer '{k}': {old_s} -> {new_s}", flush=True)
    if residual_keys:
        for k, msg in residual_keys:
            print(f"[Transfer] TeamResidualHead '{k}': {msg}", flush=True)
    print(f"[Transfer] Re-initialized {model.n_reward_channels} orthogonal channel queries & identity FiLM.", flush=True)