from pathlib import Path
import torch
import torch.nn as nn

EXCLUDED_PREFIXES = (
    "ret_ms",
    "coord_detector",
    "target_critic",  # Synced automatically via soft_update(1.0)
)


def save_transferable_backbone(model: nn.Module, path: str):
    """
    Saves policy, cross-attention, shared FiLM projections,
    universal channel embedding bank, and coordination scores.
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
    Robust universal transfer loader:
    1. Exact match for internal Transformer, Actor, and FiLM layers.
    2. Input layer zero-padding/slicing when observation dimensions shift across grid sizes.
    3. Universal slot copy for channel embeddings, readout weights, and basis coordination scores.
    """
    pretrained_dict = torch.load(path, map_location=device, weights_only=True)
    model_dict = model.state_dict()

    loaded_keys = []
    adapted_keys = []

    for k, v in pretrained_dict.items():
        if any(k.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue

        if k not in model_dict:
            continue

        target_shape = model_dict[k].shape

        # Case 1: Exact Shape Match
        if v.shape == target_shape:
            model_dict[k] = v
            loaded_keys.append(k)

        # Case 2: Observation Dimension Mismatch (e.g. Actor / Critic state input layers)
        elif len(v.shape) == 2 and v.shape[0] == target_shape[0] and v.shape[1] != target_shape[1]:
            common_in = min(v.shape[1], target_shape[1])
            model_dict[k].zero_()
            model_dict[k][:, :common_in] = v[:, :common_in]
            adapted_keys.append((k, f"Input adapted {v.shape[1]} -> {target_shape[1]}"))

        # Case 3: Universal Slot Embeddings & Readouts (w_team, b_team, embeddings, basis_coord_scores)
        elif any(sub in k for sub in ["channel_embeddings.weight", "head.w_team", "head.b_team", "basis_coord_scores"]):
            common_slots = min(v.shape[0], target_shape[0])
            model_dict[k][:common_slots] = v[:common_slots]
            adapted_keys.append((k, f"Preserved {common_slots} basis slots"))

        # Case 4: Agent Residual Weights (delta_w: N x M x d_model)
        elif "head.delta_w" in k and len(v.shape) == 3 and v.shape[2] == target_shape[2]:
            common_n = min(v.shape[0], target_shape[0])
            common_slots = min(v.shape[1], target_shape[1])
            model_dict[k][:common_n, :common_slots, :] = v[:common_n, :common_slots, :]
            adapted_keys.append((k, f"Preserved delta_w ({common_n} agents, {common_slots} slots)"))

        # Case 5: Agent Residual Biases (delta_b: N x M)
        elif "head.delta_b" in k and len(v.shape) == 2:
            common_n = min(v.shape[0], target_shape[0])
            common_slots = min(v.shape[1], target_shape[1])
            model_dict[k][:common_n, :common_slots] = v[:common_n, :common_slots]
            adapted_keys.append((k, f"Preserved delta_b ({common_n} agents, {common_slots} slots)"))

    model.load_state_dict(model_dict, strict=False)
    model.soft_update(1.0)

    print(f"[Transfer] Loaded {len(loaded_keys)} exact tensors.", flush=True)
    for k, msg in adapted_keys:
        print(f"[Transfer] {k}: {msg}", flush=True)