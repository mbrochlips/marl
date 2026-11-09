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
    universal channel embedding bank, basis coordination scores, and readout head.
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
    3. Universal slot preservation for channel query embeddings and basis coordination scores.
    4. Dynamic Agent Readout Expansion: If pretrained weights have fewer agents, new agent 
       rows in head.weight and head.bias are initialized as the median of the existing agents.
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

        # ------------------------------------------------------------------
        # Case 1: Exact Shape Match
        # ------------------------------------------------------------------
        if v.shape == target_shape:
            model_dict[k] = v
            loaded_keys.append(k)

        # ------------------------------------------------------------------
        # Case 2: State Input Dimension Mismatch (e.g. 6x6 -> 8x8 or more entities)
        # ------------------------------------------------------------------
        elif len(v.shape) == 2 and v.shape[0] == target_shape[0] and v.shape[1] != target_shape[1]:
            common_in = min(v.shape[1], target_shape[1])
            model_dict[k].zero_()
            model_dict[k][:, :common_in] = v[:, :common_in]
            adapted_keys.append((k, f"Input adapted {v.shape[1]} -> {target_shape[1]}"))

        # ------------------------------------------------------------------
        # Case 3: Universal Basis Slots (channel_embeddings.weight, basis_coord_scores)
        # ------------------------------------------------------------------
        elif any(sub in k for sub in ["channel_embeddings.weight", "basis_coord_scores"]):
            common_slots = min(v.shape[0], target_shape[0])
            model_dict[k][:common_slots] = v[:common_slots]
            adapted_keys.append((k, f"Preserved {common_slots} basis slots ({v.shape[0]} -> {target_shape[0]})"))

        # ------------------------------------------------------------------
        # Case 4: Critic Readout Head Weight: shape (n_agents, d_model)
        # ------------------------------------------------------------------
        elif "head.weight" in k and len(v.shape) == 2 and v.shape[1] == target_shape[1]:
            n_src = v.shape[0]
            n_tgt = target_shape[0]

            if n_src < n_tgt:
                # 1. Copy weights for existing agents
                model_dict[k][:n_src] = v
                # 2. Compute median weight vector across existing agents (dim=0 -> shape: (d_model,))
                median_w = torch.median(v, dim=0).values
                # 3. Fill new agent rows with the median
                model_dict[k][n_src:] = median_w
                adapted_keys.append((k, f"Expanded agents {n_src} -> {n_tgt} using median weight init"))
            else:
                # If transferring from a run with more agents to fewer agents, slice
                model_dict[k] = v[:n_tgt]
                adapted_keys.append((k, f"Sliced agents {n_src} -> {n_tgt}"))

        # ------------------------------------------------------------------
        # Case 5: Critic Readout Head Bias: shape (n_agents,)
        # ------------------------------------------------------------------
        elif "head.bias" in k and len(v.shape) == 1:
            n_src = v.shape[0]
            n_tgt = target_shape[0]

            if n_src < n_tgt:
                # 1. Copy biases for existing agents
                model_dict[k][:n_src] = v
                # 2. Compute median scalar bias across existing agents
                median_b = torch.median(v, dim=0).values
                # 3. Fill new agent entries with the median
                model_dict[k][n_src:] = median_b
                adapted_keys.append((k, f"Expanded bias {n_src} -> {n_tgt} using median bias init"))
            else:
                model_dict[k] = v[:n_tgt]
                adapted_keys.append((k, f"Sliced bias {n_src} -> {n_tgt}"))

    # Load adapted dictionary into model
    model.load_state_dict(model_dict, strict=False)
    # Sync target critic parameters immediately
    model.soft_update(1.0)

    print(f"[Transfer] Loaded {len(loaded_keys)} exact tensors.", flush=True)
    for k, msg in adapted_keys:
        print(f"[Transfer] {k}: {msg}", flush=True)