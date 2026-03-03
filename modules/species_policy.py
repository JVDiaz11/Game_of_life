from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging
import torch
import torch.nn as nn

from .ppo_buffer import PPOBuffer, PPOBufferConfig
from .ppo_trainer import PPOTrainer, PPOTrainerConfig

# Use the app logger so entries land in debug.txt.
logger = logging.getLogger("gol.policy")
logger.setLevel(logging.DEBUG)

NUM_STATES = 7  # barrier, empty, species1..5
PATCH = 5
RESOURCE_CH = 1
MEMORY_CH = 2  # keep in sync with MemoryLayer
OBS_DIM = PATCH * PATCH * (NUM_STATES + RESOURCE_CH + MEMORY_CH)
# Action mapping:
# 0 stay (keep current)
# 1 die (empty)
# 2 move up
# 3 move down
# 4 move left
# 5 move right
# 6 claim (set to own species if empty)
NUM_ACTIONS = 7

SPECIES_DIR = Path(__file__).resolve().parent.parent / "species"
SPECIES_DIR.mkdir(parents=True, exist_ok=True)


class PPOPolicyNet(nn.Module):
    def __init__(self, hidden_layers: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_features = OBS_DIM
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_features, NUM_ACTIONS)
        self.value_head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value


@dataclass
class SpeciesPolicyConfig:
    hidden_layers: Optional[List[int]] = None  # set in __post_init__
    hidden_increment: int = 128
    hidden_max: int = 1024
    mutate_every: int = 200  # steps alive before mutate attempt
    noise_std: float = 0.01

    def __post_init__(self) -> None:
        if self.hidden_layers is None:
            # Deep default architecture
            self.hidden_layers = [512, 512, 256]


@dataclass
class SpeciesPolicyManager:
    models: Dict[int, PPOPolicyNet]
    device: torch.device
    buffers: Dict[int, PPOBuffer]
    trainers: Dict[int, PPOTrainer]
    buffer_cfg: PPOBufferConfig
    trainer_cfg: PPOTrainerConfig
    trainer_cfgs: Dict[int, PPOTrainerConfig]
    species_cfg: SpeciesPolicyConfig
    species_cfgs: Dict[int, SpeciesPolicyConfig]
    species_age: Dict[int, int]
    debug_stats: Dict[int, dict]
    last_update: Dict[int, dict]
    buffer_logged: Dict[int, int]
    device_logged: bool = False

    @staticmethod
    def _arch_path(species: int) -> Path:
        return SPECIES_DIR / f"species{species}_arch.json"

    @staticmethod
    def _spawn_arch_path(species: int) -> Path:
        return SPECIES_DIR / f"species{species}_spawn.json"

    @classmethod
    def _reset_spawn_files(cls, species_ids: List[int]) -> None:
        for sid in species_ids:
            path = cls._spawn_arch_path(sid)
            if path.exists():
                path.unlink()

    @classmethod
    def _load_or_create_arch(
        cls,
        species: int,
        default_layers: List[int],
        default_trainer_cfg: PPOTrainerConfig,
        default_species_cfg: SpeciesPolicyConfig,
    ) -> Tuple[List[int], PPOTrainerConfig, SpeciesPolicyConfig]:
        path = cls._arch_path(species)
        layers: List[int] = list(default_layers)
        trainer_cfg = default_trainer_cfg
        species_cfg = default_species_cfg
        if not path.exists():
            raise FileNotFoundError(f"Architecture file not found for species {species}: {path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "hidden_layers" in data:
                layers = [int(x) for x in data["hidden_layers"]]
            elif "hidden" in data:
                layers = [int(data["hidden"])]
            if "trainer" in data and isinstance(data["trainer"], dict):
                trainer_cfg = _trainer_cfg_from_dict(data["trainer"], default_trainer_cfg)
            if "species" in data and isinstance(data["species"], dict):
                species_cfg = _species_cfg_from_dict(data["species"], default_species_cfg)
            return layers, trainer_cfg, species_cfg
        except Exception as exc:
            logger.error("Failed to read arch file", extra={"species": species, "path": str(path), "error": str(exc)})
            raise

    @classmethod
    def _write_spawn_arch(cls, species: int, hidden_layers: List[int], trainer_cfg: PPOTrainerConfig, species_cfg: SpeciesPolicyConfig) -> None:
        path = cls._spawn_arch_path(species)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "hidden_layers": [int(x) for x in hidden_layers],
                    "trainer": _serialize_trainer_cfg(trainer_cfg),
                    "species": _serialize_species_cfg(species_cfg),
                },
                f,
                indent=2,
            )

    @classmethod
    def create(
        cls,
        species_ids: List[int],
        hidden: int = 128,
        hidden_layers: Optional[List[int]] = None,
        device: Optional[str] = None,
        buffer_cfg: Optional[PPOBufferConfig] = None,
        trainer_cfg: Optional[PPOTrainerConfig] = None,
        species_cfg: Optional[SpeciesPolicyConfig] = None,
        reset_arch: bool = False,
    ) -> "SpeciesPolicyManager":
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        bcfg = buffer_cfg or PPOBufferConfig()
        tcfg = trainer_cfg or PPOTrainerConfig()
        scfg = species_cfg or SpeciesPolicyConfig(hidden_layers=hidden_layers or [hidden])
        if reset_arch:
            cls._reset_spawn_files(species_ids)
        models: Dict[int, PPOPolicyNet] = {}
        buffers: Dict[int, PPOBuffer] = {}
        trainers: Dict[int, PPOTrainer] = {}
        trainer_cfgs: Dict[int, PPOTrainerConfig] = {}
        species_cfgs: Dict[int, SpeciesPolicyConfig] = {}
        for sid in species_ids:
            layers, t_cfg, s_cfg = cls._load_or_create_arch(sid, scfg.hidden_layers, tcfg, scfg)
            trainer_cfgs[sid] = t_cfg
            species_cfgs[sid] = s_cfg
            models[sid] = PPOPolicyNet(layers).to(dev)
            buffers[sid] = PPOBuffer(OBS_DIM, bcfg, dev)
            trainers[sid] = PPOTrainer(models[sid], t_cfg, dev)
            cls._write_spawn_arch(sid, layers, t_cfg, s_cfg)
        ages = {sid: 0 for sid in species_ids}
        return cls(
            models=models,
            device=dev,
            buffers=buffers,
            trainers=trainers,
            buffer_cfg=bcfg,
            trainer_cfg=tcfg,
            trainer_cfgs=trainer_cfgs,
            species_cfg=scfg,
            species_cfgs=species_cfgs,
            species_age=ages,
            debug_stats={},
            last_update={},
            buffer_logged={},
        )

    def _ensure_species(self, species: int) -> None:
        if species in self.models:
            return
        layers, t_cfg, s_cfg = self._load_or_create_arch(species, self.species_cfg.hidden_layers, self.trainer_cfg, self.species_cfg)
        model = PPOPolicyNet(layers).to(self.device)
        self.models[species] = model
        self.buffers[species] = PPOBuffer(OBS_DIM, self.buffer_cfg, self.device)
        self.trainers[species] = PPOTrainer(model, t_cfg, self.device)
        self.trainer_cfgs[species] = t_cfg
        self.species_cfgs[species] = s_cfg
        self.species_age[species] = 0
        self._write_spawn_arch(species, layers, t_cfg, s_cfg)

    def _maybe_mutate(self, species: int) -> None:
        cfg = self.species_cfgs.get(species, self.species_cfg)
        age = self.species_age.get(species, 0)
        if age == 0 or age % cfg.mutate_every != 0:
            return
        model = self.models[species]
        src_linears = [m for m in model.trunk if isinstance(m, nn.Linear)]
        current_layers = [layer.out_features for layer in src_linears]
        new_layers = [min(h + cfg.hidden_increment, cfg.hidden_max) for h in current_layers]
        if new_layers != current_layers:
            widened = PPOPolicyNet(new_layers).to(self.device)
            dst_linears = [m for m in widened.trunk if isinstance(m, nn.Linear)]
            _pad_and_copy_mlp(src_linears, dst_linears)
            # copy heads
            _pad_and_copy_mlp([model.policy_head], [widened.policy_head])
            _pad_and_copy_mlp([model.value_head], [widened.value_head])
            self.models[species] = widened
            tcfg = self.trainer_cfgs.get(species, self.trainer_cfg)
            self.trainers[species] = PPOTrainer(widened, tcfg, self.device)
            self.trainer_cfgs[species] = tcfg
            self._write_spawn_arch(species, new_layers, tcfg, cfg)
        with torch.no_grad():
            for p in self.models[species].parameters():
                p.add_(torch.randn_like(p) * cfg.noise_std)
        self._save_species(species)

    def _save_species(self, species: int) -> None:
        path = SPECIES_DIR / f"species{species}.pt"
        torch.save(self.models[species].state_dict(), path)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, center_species: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device)
        center_species = center_species.to(self.device)
        effective_species = center_species  # 0 or negative means no species
        actions = torch.zeros(center_species.shape[0], device=self.device, dtype=torch.long)
        logprobs = torch.zeros_like(actions, dtype=torch.float)
        values = torch.zeros_like(actions, dtype=torch.float)

        if not self.device_logged and self.models:
            sample_model = next(iter(self.models.values()))
            first_linear = next((m for m in sample_model.trunk if isinstance(m, nn.Linear)), None)
            logger.debug(
                "Policy device | device=%s | cuda_available=%s | model_device=%s",
                self.device,
                torch.cuda.is_available(),
                first_linear.weight.device if first_linear is not None else self.device,
            )
            self.device_logged = True

        for sp_val in torch.unique(effective_species):
            sid = int(sp_val.item())
            if sid <= 0:
                continue
            self._ensure_species(sid)
            mask = effective_species == sid
            if not mask.any():
                continue
            logits, value = self.models[sid](obs[mask])
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample() if train else torch.argmax(logits, dim=-1)
            actions[mask] = action
            logprobs[mask] = dist.log_prob(action)
            values[mask] = value
            self.species_age[sid] = self.species_age.get(sid, 0) + 1
            self._maybe_mutate(sid)

            with torch.no_grad():
                act_counts = torch.bincount(action, minlength=NUM_ACTIONS).float()
                act_probs = (act_counts / act_counts.sum().clamp(min=1.0)).cpu().numpy()
                stats = self.debug_stats.setdefault(sid, {})
                src_linears = [m for m in self.models[sid].trunk if isinstance(m, nn.Linear)]
                current_layers = [layer.out_features for layer in src_linears]
                trainer_cfg = self.trainer_cfgs.get(sid, self.trainer_cfg)
                species_cfg = self.species_cfgs.get(sid, self.species_cfg)
                stats.update(
                    {
                        "hidden": current_layers[0] if current_layers else 0,
                        "hidden_layers": current_layers,
                        "optimizer": trainer_cfg.optimizer,
                        "lr": trainer_cfg.lr,
                        "mutate_every": species_cfg.mutate_every,
                        "param_count": sum(p.numel() for p in self.models[sid].parameters()),
                        "logits_mean": float(logits.mean()),
                        "logits_std": float(logits.std()),
                        "value_mean": float(value.mean()),
                        "value_std": float(value.std()),
                        "action_dist": act_probs,
                        "buffer_fill": self.buffers[sid].size() / max(1, self.buffer_cfg.capacity),
                    }
                )

        return actions.cpu(), logprobs.cpu(), values.cpu()

    def store_step(
        self,
        obs: torch.Tensor,
        center_species: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        obs = obs.cpu()
        center_species = center_species.cpu()
        actions = actions.cpu()
        logprobs = logprobs.cpu()
        values = values.cpu()
        rewards = rewards.cpu()
        done = done.cpu()
        effective_species = torch.where(center_species <= 0, torch.ones_like(center_species), center_species)
        for sp_val in torch.unique(effective_species):
            sid = int(sp_val.item())
            if sid <= 0:
                continue
            self._ensure_species(sid)
            mask = effective_species == sid
            if not mask.any():
                continue
            self.buffers[sid].add(
                obs[mask],
                actions[mask],
                logprobs[mask],
                values[mask],
                rewards[mask],
                done[mask],
            )

    def maybe_update(self) -> Dict[int, dict]:
        stats: Dict[int, dict] = {}
        for sid, buffer in self.buffers.items():
            size = buffer.size()
            last_log = self.buffer_logged.get(sid, -1)
            # Log fill progress each ~512 samples to avoid log spam.
            if size // 512 > last_log:
                logger.debug("Buffer fill | species=%s | size=%d | ready=%s", sid, size, buffer.ready())
                self.buffer_logged[sid] = size // 512
            if not buffer.ready():
                continue
            buffer.finalize(torch.zeros(1, device=self.device))
            batch = buffer.get()
            stats[sid] = self.trainers[sid].update(batch)
            self.last_update[sid] = stats[sid]
            logger.debug(
                "PPO update | species=%s | device=%s | batch_obs=%d",
                sid,
                self.device,
                batch["obs"].shape[0],
            )
        return stats

    def debug_snapshot(self) -> Dict[int, dict]:
        snap: Dict[int, dict] = {}
        for sid in set(list(self.models.keys()) + list(self.debug_stats.keys())):
            entry = dict(self.debug_stats.get(sid, {}))
            if sid in self.last_update:
                entry["last_update"] = self.last_update[sid]
            snap[sid] = entry
        return snap


def _pad_and_copy_mlp(src_layers: List[nn.Linear], dst_layers: List[nn.Linear]) -> None:
    for s_layer, d_layer in zip(src_layers, dst_layers):
        with torch.no_grad():
            rows = min(d_layer.out_features, s_layer.out_features)
            trainer_cfgs=trainer_cfgs,
            cols = min(d_layer.in_features, s_layer.in_features)
            species_cfgs=species_cfgs,
            d_layer.weight[:rows, :cols].copy_(s_layer.weight[:rows, :cols])
            if s_layer.bias is not None and d_layer.bias is not None:
                d_layer.bias[:rows].copy_(s_layer.bias[:rows])


def _serialize_trainer_cfg(cfg: PPOTrainerConfig) -> Dict[str, float | int | str | list]:
    return {
        "clip_eps": cfg.clip_eps,
        "entropy_coef": cfg.entropy_coef,
        "value_coef": cfg.value_coef,
        "lr": cfg.lr,
        "betas": list(cfg.betas),
        "weight_decay": cfg.weight_decay,
        "eps": cfg.eps,
        "optimizer": cfg.optimizer,
        "update_epochs": cfg.update_epochs,
        "batch_size": cfg.batch_size,
        "mini_batch_size": cfg.mini_batch_size,
        "max_grad_norm": cfg.max_grad_norm,
    }


def _trainer_cfg_from_dict(data: Dict[str, object], default: PPOTrainerConfig) -> PPOTrainerConfig:
    # start from default and override if present
    cfg = PPOTrainerConfig(
        clip_eps=float(data.get("clip_eps", default.clip_eps)),
        entropy_coef=float(data.get("entropy_coef", default.entropy_coef)),
        value_coef=float(data.get("value_coef", default.value_coef)),
        lr=float(data.get("lr", default.lr)),
        betas=tuple(data.get("betas", list(default.betas))),
        weight_decay=float(data.get("weight_decay", default.weight_decay)),
        eps=float(data.get("eps", default.eps)),
        optimizer=str(data.get("optimizer", default.optimizer)),
        update_epochs=int(data.get("update_epochs", default.update_epochs)),
        batch_size=int(data.get("batch_size", default.batch_size)),
        mini_batch_size=int(data.get("mini_batch_size", default.mini_batch_size)),
        max_grad_norm=float(data.get("max_grad_norm", default.max_grad_norm)),
    )
    return cfg


def _serialize_species_cfg(cfg: SpeciesPolicyConfig) -> Dict[str, float | int | list]:
    return {
        "hidden_layers": [int(x) for x in cfg.hidden_layers] if cfg.hidden_layers is not None else None,
        "hidden_increment": cfg.hidden_increment,
        "hidden_max": cfg.hidden_max,
        "mutate_every": cfg.mutate_every,
        "noise_std": cfg.noise_std,
    }


def _species_cfg_from_dict(data: Dict[str, object], default: SpeciesPolicyConfig) -> SpeciesPolicyConfig:
    return SpeciesPolicyConfig(
        hidden_layers=[int(x) for x in data.get("hidden_layers", default.hidden_layers or [])]
        if data.get("hidden_layers") is not None
        else default.hidden_layers,
        hidden_increment=int(data.get("hidden_increment", default.hidden_increment)),
        hidden_max=int(data.get("hidden_max", default.hidden_max)),
        mutate_every=int(data.get("mutate_every", default.mutate_every)),
        noise_std=float(data.get("noise_std", default.noise_std)),
    )
