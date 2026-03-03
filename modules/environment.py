from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict, Tuple

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from .memory import MemoryLayer, MemoryConfig
from .resources import ResourceLayer, ResourceConfig
from .resource_map import ResourceMapManager
from .species_policy import SpeciesPolicyManager, NUM_STATES, PATCH

logger = logging.getLogger("gol")


def _log(msg: str, **data: object) -> None:
    if logger.handlers:
        if data:
            parts = [f"{k}={v}" for k, v in data.items()]
            logger.debug(f"{msg} | " + ", ".join(parts))
        else:
            logger.debug(msg)


@dataclass
class EnvironmentConfig:
    species_count: int = 5
    predation_threshold: int = 3
    survival_counts: Tuple[int, int] = (2, 3)
    birth_neighbor_min: int = 3
    predator_map: Tuple[int, ...] = (0, 5, 1, 2, 3, 4)  # cyclic dominance
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    upkeep_costs: Tuple[float, ...] = (0.35, 0.35, 0.35, 0.35, 0.35)
    birth_costs: Tuple[float, ...] = (0.75, 0.75, 0.75, 0.75, 0.75)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    energy_decay: float = 0.1
    energy_conversion: float = 1.0
    stay_cost: float = 0.05
    takeover_cost: float = 0.25
    birth_action_cost: float = 0.4
    energy_kill_gain: float = 0.5


class EnvironmentEngine:
    """Applies ecological rules, resource constraints, and optional policies."""

    def __init__(self, rows: int, cols: int, config: EnvironmentConfig | None = None) -> None:
        self.cfg = config or EnvironmentConfig()
        self.resource_map = ResourceMapManager.create(rows, cols, self.cfg.resource)
        self.resources = self.resource_map.layer
        self.memory = MemoryLayer(rows, cols, self.cfg.memory)
        self.energy = np.zeros((rows, cols), dtype=np.float32)
        _log("EnvironmentEngine initialized", rows=rows, cols=cols)

    def reset(self) -> None:
        self.resource_map.reset()
        self.memory.reset()
        self.energy[:] = 0.0
        _log("Environment reset")

    def step(
        self,
        grid: np.ndarray,
        wrap: bool,
        policy_mgr: SpeciesPolicyManager | None = None,
        use_policy: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        g_prev = grid
        barriers = g_prev == -1
        energy_prev = self.energy.copy()
        self.resources.regenerate()
        self.memory.decay()
        self.energy -= self.cfg.energy_decay
        _log("Environment step start", wrap=wrap, alive=int((g_prev > 0).sum()), energy_mean=float(self.energy.mean()))

        if use_policy and policy_mgr is not None:
            obs, center_species = self._build_observations(g_prev, wrap)
            if not getattr(self, "_policy_seen", False):
                _log("Policy step engaged", obs_count=int(obs.shape[0]))
                self._policy_seen = True
            actions, logprobs, values = policy_mgr.act(obs, center_species, train=True)
            actions_np = actions.view(g_prev.shape).cpu().numpy()
            candidate, self.energy = self._apply_actions(g_prev, self.energy, actions_np, wrap)
        else:
            obs = None
            center_species = None
            logprobs = None
            values = None
            candidate = self._rule_step(g_prev, wrap)

        births = (candidate > 0) & (g_prev <= 0)
        shortages = np.zeros_like(candidate, dtype=bool)
        food_ok = np.zeros_like(candidate, dtype=bool)

        for species in range(1, self.cfg.species_count + 1):
            birth_mask = births & (candidate == species)
            birth_cost = self._birth_cost(species)
            starved_birth = self.resources.consume(birth_mask, birth_cost)
            feasible_birth = birth_mask & (~starved_birth)
            candidate[starved_birth] = 0

            upkeep_mask = (candidate == species) & (~birth_mask)
            upkeep_cost = self._upkeep_cost(species)
            starved_upkeep = self.resources.consume(upkeep_mask, upkeep_cost)
            feasible_upkeep = upkeep_mask & (~starved_upkeep)
            candidate[starved_upkeep] = 0
            shortages |= starved_birth | starved_upkeep
            food_ok |= feasible_birth | feasible_upkeep

            self.energy[feasible_birth] += birth_cost * self.cfg.energy_conversion
            self.energy[feasible_upkeep] += upkeep_cost * self.cfg.energy_conversion

        # action energy costs and death if depleted
        stay_mask = (candidate == g_prev) & (candidate > 0)
        takeover = (g_prev > 0) & (candidate > 0) & (candidate != g_prev)
        births_final = (candidate > 0) & (g_prev <= 0)

        if np.any(takeover):
            self.energy[takeover] += energy_prev[takeover] * self.cfg.energy_kill_gain

        self.energy[stay_mask] -= self.cfg.stay_cost
        self.energy[takeover] -= self.cfg.takeover_cost
        self.energy[births_final] -= self.cfg.birth_action_cost

        died_energy = (self.energy < 0) & (candidate > 0)
        candidate[died_energy] = 0
        self.energy[died_energy] = 0.0

        reward_grid = self._compute_rewards(g_prev, candidate, shortages)
        self.memory.set_reward(reward_grid)
        self.memory.reset_food_timer(food_ok)

        info: Dict[str, float] = {
            "births": float(births.sum()),
            "starved": float(shortages.sum()),
            "resources_avg": float(self.resources.grid.mean()),
            "reward_mean": float(reward_grid.mean()),
        }

        if use_policy and policy_mgr is not None and obs is not None and center_species is not None:
            rewards_flat = torch.from_numpy(reward_grid.reshape(-1).astype(np.float32))
            done = torch.zeros_like(rewards_flat)
            policy_mgr.store_step(obs, center_species, actions, logprobs, values, rewards_flat, done)
            update_stats = policy_mgr.maybe_update()
            if update_stats:
                info["ppo_updates"] = len(update_stats)

        _log(
            "Environment step end",
            births=info.get("births", 0.0),
            starved=info.get("starved", 0.0),
            resources_avg=info.get("resources_avg", 0.0),
            reward_mean=info.get("reward_mean", 0.0),
            energy_mean=float(self.energy.mean()),
        )
        return candidate.astype(np.int8), info

    def _apply_actions(self, g_prev: np.ndarray, energy: np.ndarray, actions_np: np.ndarray, wrap: bool) -> tuple[np.ndarray, np.ndarray]:
        # action codes defined in species_policy.NUM_ACTIONS
        # 0 stay, 1 die, 2 up, 3 down, 4 left, 5 right, 6 claim
        candidate = g_prev.copy()
        energy_new = energy.copy()

        alive_mask = g_prev > 0

        # die
        die_mask = (actions_np == 1) & alive_mask
        candidate[die_mask] = 0
        energy_new[die_mask] = 0.0

        def move(dir_code: int, dr: int, dc: int) -> None:
            src = (actions_np == dir_code) & alive_mask
            if not src.any():
                return

            r_idx, c_idx = np.nonzero(src)

            if not wrap:
                valid = (
                    (r_idx + dr >= 0)
                    & (r_idx + dr < g_prev.shape[0])
                    & (c_idx + dc >= 0)
                    & (c_idx + dc < g_prev.shape[1])
                )
                if not valid.any():
                    return
                r_idx = r_idx[valid]
                c_idx = c_idx[valid]

            tr = (r_idx + dr) % g_prev.shape[0]
            tc = (c_idx + dc) % g_prev.shape[1]

            for r, c, nr, nc in zip(r_idx, c_idx, tr, tc):
                if g_prev[nr, nc] == -1:
                    continue  # barrier
                if candidate[nr, nc] == 0:
                    candidate[nr, nc] = g_prev[r, c]
                    energy_new[nr, nc] = energy_new[r, c]
                    candidate[r, c] = 0
                    energy_new[r, c] = 0.0
                elif candidate[nr, nc] != g_prev[r, c]:
                    candidate[nr, nc] = g_prev[r, c]  # takeover
                    energy_new[nr, nc] = energy_new[r, c]
                    candidate[r, c] = 0
                    energy_new[r, c] = 0.0

        move(2, -1, 0)
        move(3, 1, 0)
        move(4, 0, -1)
        move(5, 0, 1)

        # claim empty cell (use own species if alive)
        species_for_claim = np.where(g_prev > 0, g_prev, 1)
        claim_mask = (actions_np == 6) & (candidate <= 0)
        candidate[claim_mask] = species_for_claim[claim_mask]

        return candidate, energy_new

    def _birth_cost(self, species: int) -> float:
        idx = min(species - 1, len(self.cfg.birth_costs) - 1)
        return self.cfg.birth_costs[idx]

    def _upkeep_cost(self, species: int) -> float:
        idx = min(species - 1, len(self.cfg.upkeep_costs) - 1)
        return self.cfg.upkeep_costs[idx]

    def _rule_step(self, g: np.ndarray, wrap: bool) -> np.ndarray:
        neighbor_counts = []
        for s in range(1, self.cfg.species_count + 1):
            mask = g == s
            if wrap:
                n = sum(
                    np.roll(np.roll(mask, i, axis=0), j, axis=1)
                    for i in (-1, 0, 1)
                    for j in (-1, 0, 1)
                    if not (i == 0 and j == 0)
                )
            else:
                n = np.zeros_like(mask, dtype=np.int8)
                n[1:-1, 1:-1] = (
                    mask[:-2, :-2]
                    + mask[:-2, 1:-1]
                    + mask[:-2, 2:]
                    + mask[1:-1, :-2]
                    + mask[1:-1, 2:]
                    + mask[2:, :-2]
                    + mask[2:, 1:-1]
                    + mask[2:, 2:]
                )
            neighbor_counts.append(n.astype(np.int16))
        counts_stack = np.stack(neighbor_counts, axis=-1)

        new_grid = np.zeros_like(g)
        barrier_mask = g == -1
        new_grid[barrier_mask] = -1

        pred_map = np.array(self.cfg.predator_map, dtype=np.int8)
        species_mask = g > 0
        predators = pred_map[g.clip(0, self.cfg.species_count)]
        flat_counts = counts_stack.reshape(-1, self.cfg.species_count)
        pred_idx = np.clip(predators.reshape(-1) - 1, 0, self.cfg.species_count - 1)
        pred_counts = flat_counts[np.arange(flat_counts.shape[0]), pred_idx].reshape(g.shape)
        predation_mask = species_mask & (pred_counts >= self.cfg.predation_threshold)

        for s in range(1, self.cfg.species_count + 1):
            s_mask = g == s
            same = counts_stack[..., s - 1]
            survive = s_mask & (~predation_mask) & (
                (same == self.cfg.survival_counts[0]) | (same == self.cfg.survival_counts[1])
            )
            new_grid[survive] = s

        new_grid[predation_mask] = predators[predation_mask]

        empty_mask = g == 0
        max_count = counts_stack.max(axis=-1)
        argmax = counts_stack.argmax(axis=-1) + 1
        tie_mask = (counts_stack == max_count[..., None]).sum(axis=-1) > 1
        birth_mask = empty_mask & (max_count >= self.cfg.birth_neighbor_min) & (~tie_mask)
        new_grid[birth_mask] = argmax[birth_mask]
        return new_grid

    def _build_observations(self, grid: np.ndarray, wrap: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized observation builder to cut per-step Python overhead."""
        H, W = grid.shape
        pad = PATCH // 2

        if wrap:
            state_pad = np.pad(grid, pad, mode="wrap")
        else:
            state_pad = np.pad(grid, pad, mode="constant", constant_values=0)
        res_pad = np.pad(self.resources.as_array(), pad, mode="wrap" if wrap else "edge")
        mem_pad = np.pad(self.memory.as_array(), ((pad, pad), (pad, pad), (0, 0)), mode="wrap" if wrap else "edge")

        state_win = sliding_window_view(state_pad, (PATCH, PATCH))  # (H, W, PATCH, PATCH)
        res_win = sliding_window_view(res_pad, (PATCH, PATCH))
        mem_win = sliding_window_view(mem_pad, (PATCH, PATCH), axis=(0, 1))

        mapped = np.clip(state_win + 1, 0, NUM_STATES - 1)
        oh = np.eye(NUM_STATES, dtype=np.float32)[mapped]  # (H, W, PATCH, PATCH, NUM_STATES)

        obs = np.concatenate(
            [
                oh.reshape(H, W, -1),
                res_win.reshape(H, W, -1).astype(np.float32),
                mem_win.reshape(H, W, -1).astype(np.float32),
            ],
            axis=-1,
        ).reshape(-1, PATCH * PATCH * (NUM_STATES + self.memory.cfg.channels + 1))

        center_species = np.where(grid <= 0, 0, grid).reshape(-1).astype(np.int64)

        return torch.from_numpy(obs).float(), torch.from_numpy(center_species)

    def _compute_rewards(self, prev: np.ndarray, new: np.ndarray, shortages: np.ndarray) -> np.ndarray:
        reward = np.zeros_like(prev, dtype=np.float32)
        alive_prev = prev > 0
        alive_new = new > 0

        survive = alive_prev & (new == prev)
        reward[survive] += 1.0

        births = (~alive_prev) & alive_new
        reward[births] += 2.0

        death = alive_prev & (~alive_new)
        reward[death] -= 2.0

        takeover = alive_prev & alive_new & (new != prev)
        reward[takeover] += 1.0

        reward[shortages] -= 1.0
        return reward
