from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn, optim


@dataclass
class PPOTrainerConfig:
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    eps: float = 1e-8
    optimizer: str = "adam"  # "adam" or "adamw"
    update_epochs: int = 3
    batch_size: int = 1024
    mini_batch_size: int = 256
    max_grad_norm: float = 1.0


class PPOTrainer:
    def __init__(self, model: nn.Module, cfg: PPOTrainerConfig | None = None, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.cfg = cfg or PPOTrainerConfig()
        self.device = device or torch.device("cpu")
        if self.cfg.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.lr,
                betas=self.cfg.betas,
                weight_decay=self.cfg.weight_decay,
                eps=self.cfg.eps,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.cfg.lr,
                betas=self.cfg.betas,
                weight_decay=self.cfg.weight_decay,
                eps=self.cfg.eps,
            )

    def update(self, batch: dict) -> dict:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device).long()
        old_logprobs = batch["logprobs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        values_old = batch["values"].to(self.device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx = torch.randperm(obs.size(0))
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        grad_steps = 0

        for _ in range(self.cfg.update_epochs):
            for start in range(0, obs.size(0), self.cfg.mini_batch_size):
                end = start + self.cfg.mini_batch_size
                mb_idx = idx[start:end]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits, values = self.model(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_logprobs - mb_old_logprobs).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = values_old[mb_idx] + (values - values_old[mb_idx]).clamp(-self.cfg.clip_eps, self.cfg.clip_eps)
                value_losses = (values - mb_returns).pow(2)
                value_losses_clipped = (value_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_grad_norm += float(grad_norm)
                grad_steps += 1

        steps = max(1, (obs.size(0) // self.cfg.mini_batch_size) * self.cfg.update_epochs)
        return {
            "policy_loss": total_policy_loss / steps,
            "value_loss": total_value_loss / steps,
            "entropy": total_entropy / steps,
            "grad_norm": (total_grad_norm / max(1, grad_steps)),
            "lr": self.optimizer.param_groups[0]["lr"],
        }
