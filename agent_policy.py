"""Per-species policy networks for the Life variant.

This is intentionally separate from the GUI/game loop to avoid bloating the main file.
Each species gets its own small MLP that maps a local patch to an action (next state).

States: -1 barrier, 0 empty, 1-5 species.
Actions: 0 empty or 1-5 species (barriers stay unchanged outside the policy).
Input encoding: one-hot over 7 states for a 5x5 patch (25*7 features).

Note: these policies are untrained by default (random weights). Train them externally
and load state_dicts into PolicyManager.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_STATES = 7  # barrier, empty, species1..5
PATCH = 5
INPUT_DIM = PATCH * PATCH * NUM_STATES
NUM_ACTIONS = 6  # empty + 5 species (barrier unaffected)


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PolicyManager:
    models: Dict[int, PolicyNet]
    device: torch.device

    @classmethod
    def create(cls, species_ids: List[int], hidden: int = 64, device: Optional[str] = None) -> "PolicyManager":
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        models = {sid: PolicyNet(hidden).to(dev) for sid in species_ids}
        return cls(models=models, device=dev)

    def load_state(self, paths: Dict[int, str]) -> None:
        for sid, p in paths.items():
            self.models[sid].load_state_dict(torch.load(p, map_location=self.device))

    @torch.no_grad()
    def infer_next_states(self, grid: torch.Tensor) -> torch.Tensor:
        """Infer next state for each cell using species-specific policies.

        grid: int8/long tensor HxW with states {-1,0,1..5}
        Returns new_grid (HxW) with states {0..5} (barriers unchanged elsewhere).
        """
        H, W = grid.shape
        pad = PATCH // 2
        # pad with zeros (empty); barriers remain only inside original grid
        padded = F.pad(grid, (pad, pad, pad, pad), mode="constant", value=0)

        # build patches
        patches = []
        for r in range(H):
            for c in range(W):
                patch = padded[r : r + PATCH, c : c + PATCH]
                patches.append(patch)
        patches_t = torch.stack(patches)  # (N, 5, 5)

        # one-hot encode 7 states: map -1->0 (barrier), 0->1 (empty), 1-5->2..6
        mapped = patches_t + 1  # barrier becomes 0, empty 1, species k -> k+1
        oh = F.one_hot(mapped.clamp(min=0, max=6), num_classes=NUM_STATES)  # (N,5,5,7)
        oh_flat = oh.view(oh.shape[0], -1).float()

        # choose model by center species; if empty, pick model 1 for symmetry
        center_states = grid.view(-1)
        center_species = torch.where(center_states <= 0, torch.ones_like(center_states), center_states)

        logits_list = []
        start = 0
        for sid, model in self.models.items():
            mask = center_species == sid
            if mask.any():
                inputs = oh_flat[mask].to(self.device)
                logits = model(inputs)
                logits_list.append((mask, logits.cpu()))
        # combine logits back
        new_states = torch.zeros_like(center_states)
        for mask, logits in logits_list:
            actions = logits.argmax(dim=-1)  # 0..5
            new_states[mask] = actions

        return new_states.view(H, W)
