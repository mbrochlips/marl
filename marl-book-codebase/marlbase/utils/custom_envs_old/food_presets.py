import copy

# ==============================================================================
# GLOBAL POOL OF FOOD TYPES (Agents are Level 4)
# Solo foods: Levels 1 - 4 (1 agent required)
# Coop foods: Levels 5 - 8 (2 agents required, combined level = 8)
# ==============================================================================

# --- Solo Foods ---
solo_det = {
    "level": 1,
    "type": "solo_det",
    "reward": 1.0,
    "has_penalty": False,
    "split": True,
}
solo_det_high = {
    "level": 2,
    "type": "solo_det_high",
    "reward": 1.5,
    "has_penalty": False,
    "split": True,
}
solo_stoch = {
    "level": 2,
    "type": "solo_stoch",
    "stoch_outcomes": [-2.0, 1.0],
    "stoch_probs": [0.5, 0.5],
    "has_penalty": False,
    "split": False,
}
solo_stoch_risky = {
    "level": 4,
    "type": "solo_stoch_risky",
    "stoch_outcomes": [-3.0, 2.0],
    "stoch_probs": [0.5, 0.5],
    "has_penalty": False,
    "split": True,
}

# --- Cooperative Foods ---
coop_det = {
    "level": 5,
    "type": "coop_det",
    "reward": 3.0,
    "has_penalty": True,
    "split": False,
}
coop_det_free = {
    "level": 6,
    "type": "coop_det_free",
    "reward": 2.0,
    "has_penalty": False,
    "split": False,
}
coop_super = {
    "level": 7,
    "type": "coop_super",
    "reward": 5.0,
    "has_penalty": True,
    "split": False,
}
coop_stoch = {
    "level": 8,
    "type": "coop_stoch",
    "stoch_outcomes": [-0.5, 3.0],
    "stoch_probs": [0.2, 0.8],
    "has_penalty": False,
    "split": False,
}


# ==============================================================================
# FOOD PRESETS
# ==============================================================================

FOOD_PRESETS = {
    "3foods": {
        "food_items": [
            coop_det,
            solo_det,
            solo_stoch,
        ]
    },
    "4foods": {
        "food_items": [
            coop_det,
            coop_det_free,
            solo_det,
            solo_stoch,
        ]
    },
    "8foods": {
        "food_items": [
            solo_det,
            solo_det_high,
            solo_stoch,
            solo_stoch_risky,
            coop_det,
            coop_det_free,
            coop_super,
            coop_stoch,
        ]
    },
}

# ==============================================================================
# DYNAMIC CHANNEL_ID & CHANNEL_NAMES GENERATION
# ==============================================================================
for preset in FOOD_PRESETS.values():
    preset["food_items"] = [{**copy.deepcopy(item), "channel_id": i} for i, item in enumerate(preset["food_items"])]
    preset["channel_names"] = [item["type"] for item in preset["food_items"]] + ["step_penalty"]