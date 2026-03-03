from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class PPOBufferConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    capacity: int = 512


class PPOBuffer:
    """Rollout storage for a single species."""

    def __init__(self, obs_dim: int, cfg: PPOBufferConfig | None = None, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg or PPOBufferConfig()
        self.device = device or torch.device("cpu")
        self.obs_dim = obs_dim
        self.reset()

    def reset(self) -> None:
        self.obs: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.last_value: Optional[torch.Tensor] = None

    def add(self, obs: torch.Tensor, action: torch.Tensor, logprob: torch.Tensor, value: torch.Tensor, reward: torch.Tensor, done: torch.Tensor) -> None:
        # obs: (N, obs_dim)
        self.obs.append(obs.detach())
        self.actions.append(action.detach())
        self.logprobs.append(logprob.detach())
        self.rewards.append(reward.detach())
        self.values.append(value.detach())
        self.dones.append(done.detach())

    def size(self) -> int:
        return sum(t.numel() for t in self.actions)

    def ready(self) -> bool:
        return self.size() >= self.cfg.capacity

    def finalize(self, last_value: torch.Tensor) -> None:
        self.last_value = last_value.detach()

    def get(self) -> dict:
        obs = torch.cat(self.obs, dim=0).to(self.device)
        actions = torch.cat(self.actions, dim=0).to(self.device)
        logprobs = torch.cat(self.logprobs, dim=0).to(self.device)
        rewards = torch.cat(self.rewards, dim=0).to(self.device)
        values = torch.cat(self.values, dim=0).to(self.device)
        dones = torch.cat(self.dones, dim=0).to(self.device)

        if self.last_value is None:
            self.last_value = torch.zeros(1, device=self.device)
        values_ext = torch.cat([values, self.last_value.expand(1)], dim=0)

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0.0
        for t in reversed(range(rewards.shape[0])):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * values_ext[t + 1] * mask - values_ext[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

        data = {
            "obs": obs,
            "actions": actions,
            "logprobs": logprobs,
            "advantages": advantages,
            "returns": returns,
            "values": values,
        }
        self.reset()
        return data
