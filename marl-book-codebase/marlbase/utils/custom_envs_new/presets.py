"""
Declarative food presets for Level-Based Foraging environments.
"""

FOOD_PRESETS = {
    "3foods": {
        "min_player_level": 2,
        "max_player_level": 2,
        "min_spawn": 3,
        "max_spawn": 3,
        "leftover_food": 1,
        "field_size": (6, 6),
        "sight": 6,
        "foods": [
            {"type": "coop_food", "level": 3, "reward": 3.0, "has_penalty": True},
            {"type": "det_food", "level": 1, "reward": 1.0, "split": True},
            {"type": "stoch_food", "level": 2, "stoch_outcomes": [-2.0, 1.0]},
        ],
    },
    "4foods": {
        "min_player_level": 2,
        "max_player_level": 2,
        "min_spawn": 4,
        "max_spawn": 4,
        "leftover_food": 1,
        "field_size": (6, 6),
        "sight": 6,
        "foods": [
            {"type": "coop_pen", "level": 3, "reward": 4.0, "has_penalty": True},
            {"type": "coop_free", "level": 3, "reward": 2.5},
            {"type": "stoch", "level": 1, "stoch_outcomes": [-2.5, 2.0]},
            {"type": "det", "level": 1, "reward": 1.2, "split": True},
        ],
    },
    "8foods": {
        "min_player_level": 4,
        "max_player_level": 4,
        "min_spawn": 3,
        "max_spawn": 4,
        "leftover_food": 1,
        "field_size": (8, 8),
        "sight": 8,
        "foods": [
            {"type": "coop_pen_high", "level": 5, "reward": 4.0, "has_penalty": True},
            {"type": "coop_free", "level": 6, "reward": 2.5},
            {"type": "coop_super", "level": 7, "reward": 5.0, "has_penalty": True},
            {"type": "solo_det_low", "level": 1, "reward": 1.0, "split": True},
            {"type": "solo_det_high", "level": 2, "reward": 2.0, "split": True},
            {"type": "solo_stoch_risky", "level": 3, "stoch_outcomes": [-2.0, 2.5]},
            {"type": "solo_stoch_safe", "level": 4, "stoch_outcomes": [0.5, 1.5]},
            {"type": "coop_stoch", "level": 8, "stoch_outcomes": [-1.0, 4.0], "has_penalty": True},
        ],
    },
}

# Auto-generate channel_names for each preset so callers/wrappers never hit KeyError
for preset in FOOD_PRESETS.values():
    if "foods" in preset:
        preset["channel_names"] = [f["type"] for f in preset["foods"]] + ["step_penalty"]
    elif "food_items" in preset:
        preset["channel_names"] = [f["type"] for f in preset["food_items"]] + ["step_penalty"]