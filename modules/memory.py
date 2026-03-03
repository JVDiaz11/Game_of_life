from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MemoryConfig:
    decay: float = 0.9
    reward_clip: float = 3.0
    channels: int = 4  # 0: pos signal, 1: neg signal, 2: time_since_food, 3: barrier signal


class MemoryLayer:
    """Lightweight per-cell memory grid.

        Channel layout:
            0: positive signal (decays each step)
            1: negative signal (decays each step)
            2: time since food (increments each step, resets on successful upkeep/birth)
            3: barrier signal (decays each step)
    """

    def __init__(self, rows: int, cols: int, config: MemoryConfig | None = None) -> None:
        self.rows = rows
        self.cols = cols
        self.cfg = config or MemoryConfig()
        self.grid = np.zeros((rows, cols, self.cfg.channels), dtype=np.float32)

    def reset(self) -> None:
        self.grid[:] = 0.0

    def decay(self) -> None:
        self.grid[..., 0] *= self.cfg.decay  # positive
        self.grid[..., 1] *= self.cfg.decay  # negative
        self.grid[..., 2] += 1.0  # time since food
        self.grid[..., 3] *= self.cfg.decay  # barrier

    def record_events(self, pos_mask: np.ndarray, neg_mask: np.ndarray, barrier_mask: np.ndarray, pos_value: float = 1.0, neg_value: float = 1.0) -> None:
        # Mark positive/negative/barrier events; signals decay each step separately.
        if pos_mask.any():
            self.grid[..., 0][pos_mask] = np.clip(pos_value, 0.0, self.cfg.reward_clip)
        if neg_mask.any():
            self.grid[..., 1][neg_mask] = np.clip(neg_value, 0.0, self.cfg.reward_clip)
        if barrier_mask.any():
            self.grid[..., 3][barrier_mask] = 1.0

    def reset_food_timer(self, mask: np.ndarray) -> None:
        # mask: locations where upkeep/birth succeeded
        self.grid[..., 2][mask] = 0.0

    def as_array(self) -> np.ndarray:
        return self.grid
