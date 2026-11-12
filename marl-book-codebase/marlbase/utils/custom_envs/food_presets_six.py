import copy

solo_det = {
    "level": 1,
    "type": "solo_det",
    "reward": 1.0,
    "has_penalty": False,
    "split": True,
}
solo_stoch = {
    "level": 2,
    "type": "solo_stoch",
    "stoch_outcomes": [-1.0, 1.0],
    "stoch_probs": [0.5, 0.5],
    "has_penalty": False,
    "split": False,
}
solo_stoch_risky = {
    "level": 3,
    "type": "solo_stoch_risky",
    "stoch_outcomes": [-3.0, 2.0],
    "stoch_probs": [0.5, 0.5],
    "has_penalty": False,
    "split": True,
}

coop_det = {
    "level": 4,
    "type": "coop_det",
    "reward": 3.0,
    "has_penalty": True,
    "split": False,
}
coop_det_free = {
    "level": 5,
    "type": "coop_det_free",
    "reward": 2.0,
    "has_penalty": False,
    "split": False,
}
coop_stoch = {
    "level": 6,
    "type": "coop_stoch",
    "stoch_outcomes": [-0.5, 3.0],
    "stoch_probs": [0.2, 0.8],
    "has_penalty": False,
    "split": False,
}

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
    "ALL": {
        "food_items": [
            solo_det,
            solo_stoch,
            solo_stoch_risky,
            coop_det,
            coop_det_free,
            coop_stoch,
        ]
    },
}

for preset in FOOD_PRESETS.values():
    preset["food_items"] = [copy.deepcopy(item) for item in preset["food_items"]]
    preset["channel_names"] = [item["type"] for item in preset["food_items"]]