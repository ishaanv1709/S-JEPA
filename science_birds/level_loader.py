"""
Level Loader — Manages level iteration for benchmarking and data collection.

Supports both online (Science Birds server) and offline (simulator) modes.
"""

import numpy as np
from typing import Iterator


class LevelLoader:
    """Iterate through levels for data collection and benchmarking."""

    def __init__(self, mode: str = "offline", n_levels: int = 50,
                 difficulties: list = None, seed: int = 42):
        """
        Args:
            mode: 'online' (Science Birds server) or 'offline' (simulator)
            n_levels: number of levels to generate/play
            difficulties: list of difficulty levels to cycle through
            seed: random seed for reproducible level generation
        """
        self.mode = mode
        self.n_levels = n_levels
        self.difficulties = difficulties or ["easy", "medium", "hard"]
        self.seed = seed

    def __len__(self):
        return self.n_levels

    def __iter__(self) -> Iterator:
        """Yield level configs one at a time."""
        if self.mode == "offline":
            yield from self._iter_offline()
        else:
            yield from self._iter_online()

    def _iter_offline(self) -> Iterator:
        """Generate levels using the offline simulator."""
        from .client import OfflineSimulator

        for i in range(self.n_levels):
            difficulty = self.difficulties[i % len(self.difficulties)]
            sim = OfflineSimulator(seed=self.seed + i)
            level = sim.generate_level(difficulty)
            yield {
                "level_id": i + 1,
                "difficulty": difficulty,
                "state": level,
                "simulator": sim,
            }

    def _iter_online(self) -> Iterator:
        """Load levels from the Science Birds server."""
        for i in range(self.n_levels):
            yield {
                "level_id": i + 1,
                "difficulty": "unknown",
            }

    def get_benchmark_levels(self, n: int = 50) -> list:
        """
        Get a fixed set of levels for reproducible benchmarking.
        Always uses the same seed for consistency.
        """
        loader = LevelLoader(
            mode=self.mode,
            n_levels=n,
            difficulties=self.difficulties,
            seed=0,  # fixed seed for benchmark
        )
        return list(loader)


if __name__ == "__main__":
    loader = LevelLoader(mode="offline", n_levels=5)
    for level_info in loader:
        state = level_info["state"]
        print(f"Level {level_info['level_id']} ({level_info['difficulty']}): "
              f"{len(state['birds'])} birds, {len(state['blocks'])} blocks, "
              f"{len(state['pigs'])} pigs")
