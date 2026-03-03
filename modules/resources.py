from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger("gol")


def _log(msg: str, **data: object) -> None:
    if logger.handlers:
        if data:
            parts = [f"{k}={v}" for k, v in data.items()]
            logger.debug(f"{msg} | " + ", ".join(parts))
        else:
            logger.debug(msg)


@dataclass
class ResourceConfig:
    regen_rate: float = 0.25
    max_capacity: float = 5.0
    baseline: float = 1.0
    upkeep_cost: float = 0.35
    birth_cost: float = 0.75
    rng_low: float = 0.5
    rng_high: float = 3.0


class ResourceLayer:
    """Tracks consumable resources ("food") for each cell in the grid."""

    def __init__(self, rows: int, cols: int, config: ResourceConfig | None = None) -> None:
        self.rows = rows
        self.cols = cols
        self.config = config or ResourceConfig()
        self.grid = np.zeros((rows, cols), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.grid[:] = self.config.baseline
        _log("Resources reset", baseline=self.config.baseline)

    def fill(self, value: float) -> None:
        self.grid[:] = np.clip(value, 0.0, self.config.max_capacity)
        _log("Resources filled", value=value)

    def populate_random(self, low: float | None = None, high: float | None = None) -> None:
        lo = low if low is not None else self.config.rng_low
        hi = high if high is not None else self.config.rng_high
        self.grid[:] = np.clip(np.random.uniform(lo, hi, size=self.grid.shape), 0.0, self.config.max_capacity)
        _log(
            "Resources randomized",
            low=lo,
            high=hi,
            rmin=float(self.grid.min()),
            rmax=float(self.grid.max()),
            rmean=float(self.grid.mean()),
        )

    def regenerate(self) -> None:
        np.minimum(self.grid + self.config.regen_rate, self.config.max_capacity, out=self.grid)
        _log("Resources regenerated", regen=self.config.regen_rate, rmean=float(self.grid.mean()))

    def consume(self, mask: np.ndarray, amount: float) -> np.ndarray:
        """Consume resources where mask is True.

        Returns a boolean mask (same shape) where consumption failed due to shortage.
        """
        shortage = (self.grid < amount) & mask
        feasible = mask & (~shortage)
        self.grid[feasible] -= amount
        self.grid[shortage] = 0.0
        _log(
            "Resources consumed",
            amount=amount,
            feasible=int(feasible.sum()),
            shortage=int(shortage.sum()),
            rmean=float(self.grid.mean()),
        )
        return shortage

    def as_array(self) -> np.ndarray:
        return self.grid
