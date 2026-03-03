from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np

from .resources import ResourceLayer, ResourceConfig

logger = logging.getLogger("gol")


def _log(msg: str, **data: object) -> None:
    if logger.handlers:
        if data:
            parts = [f"{k}={v}" for k, v in data.items()]
            logger.debug(f"{msg} | " + ", ".join(parts))
        else:
            logger.debug(msg)


@dataclass
class ResourceMapManager:
    layer: ResourceLayer
    energy_ref: Optional[np.ndarray] = None

    @classmethod
    def create(cls, rows: int, cols: int, config: ResourceConfig | None = None, energy_ref: Optional[np.ndarray] = None) -> "ResourceMapManager":
        layer = ResourceLayer(rows, cols, config or ResourceConfig())
        _log("ResourceMapManager.create", rows=rows, cols=cols)
        return cls(layer=layer, energy_ref=energy_ref)

    def reset(self) -> None:
        self.layer.reset()
        self._clear_energy()
        _log("Resource map reset")

    def randomize(self, low: float | None = None, high: float | None = None) -> None:
        self.layer.populate_random(low, high)
        self._clear_energy()
        grid = self.layer.grid
        _log("Resource map randomized", rmin=float(grid.min()), rmax=float(grid.max()), rmean=float(grid.mean()))

    def fill(self, value: float) -> None:
        self.layer.fill(value)
        self._clear_energy()
        grid = self.layer.grid
        _log("Resource map filled", value=value, rmin=float(grid.min()), rmax=float(grid.max()), rmean=float(grid.mean()))

    def _clear_energy(self) -> None:
        if self.energy_ref is not None:
            self.energy_ref[:] = 0.0
            _log("Energy reference cleared")

    @property
    def grid(self) -> np.ndarray:
        return self.layer.grid
