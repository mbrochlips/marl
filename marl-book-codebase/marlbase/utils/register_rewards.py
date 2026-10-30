import json
from pathlib import Path
from typing import List, Dict


class RewardRegistry:
    """
    Persistent mapping from semantic reward strings to universal basis indices.
    """
    def __init__(self, max_channels: int = 128):
        self.max_channels = max_channels
        self.channel_map: Dict[str, int] = {}

    def get_or_register(self, channel_names: List[str]) -> List[int]:
        """
        Takes a list of channel strings from the active environment.
        Returns the basis indices, automatically registering any brand new IDs.
        """
        indices = []
        for name in channel_names:
            if name not in self.channel_map:
                new_idx = len(self.channel_map)
                assert new_idx < self.max_channels, f"Exceeded maximum channel capacity of {self.max_channels}."
                self.channel_map[name] = new_idx
                print(f"[RewardRegistry] Registered NEW reward ID '{name}' -> Basis Slot u_{new_idx}", flush=True)
            indices.append(self.channel_map[name])
        return indices

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.channel_map, f, indent=2)

    def load(self, path: str):
        if Path(path).exists():
            with open(path, "r") as f:
                self.channel_map = json.load(f)
            print(f"[RewardRegistry] Loaded {len(self.channel_map)} known reward channels from {path}", flush=True)