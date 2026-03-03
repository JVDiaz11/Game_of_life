from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MemoryConfig:
    decay: float = 0.9
    reward_clip: float = 3.0
    channels: int = 2  # 0: last_reward, 1: time_since_food


class MemoryLayer:
    """Lightweight per-cell memory grid.

    Channel layout:
      0: last reward (decays each step)
      1: time since food (increments each step, resets on successful upkeep/birth)
    """

    def __init__(self, rows: int, cols: int, config: MemoryConfig | None = None) -> None:
        self.rows = rows
        self.cols = cols
        self.cfg = config or MemoryConfig()
        self.grid = np.zeros((rows, cols, self.cfg.channels), dtype=np.float32)

    def reset(self) -> None:
        self.grid[:] = 0.0

    def decay(self) -> None:
        self.grid[..., 0] *= self.cfg.decay  # last reward
        self.grid[..., 1] += 1.0  # time since food

    def set_reward(self, reward_grid: np.ndarray) -> None:
        clipped = np.clip(reward_grid, -self.cfg.reward_clip, self.cfg.reward_clip)
        self.grid[..., 0] = clipped

    def reset_food_timer(self, mask: np.ndarray) -> None:
        # mask: locations where upkeep/birth succeeded
        self.grid[..., 1][mask] = 0.0

    def as_array(self) -> np.ndarray:
        return self.grid
