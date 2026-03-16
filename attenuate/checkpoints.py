from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from attenuate.eval_runtime import compare_metric_dicts


def extract_state_dict(blob: Any) -> dict[str, torch.Tensor]:
    if isinstance(blob, dict):
        if all(isinstance(v, torch.Tensor) for v in blob.values()):
            return blob
        for key in ("state_dict", "model_state", "model"):
            value = blob.get(key)
            if isinstance(value, dict) and all(isinstance(v, torch.Tensor) for v in value.values()):
                return value
    raise ValueError("Unable to extract model weights from checkpoint blob")


def extract_model_config(blob: Any) -> dict[str, Any] | None:
    if not isinstance(blob, dict):
        return None
    if isinstance(blob.get("model_config"), dict):
        return copy.deepcopy(blob["model_config"])
    config = blob.get("config")
    if isinstance(config, dict) and isinstance(config.get("model"), dict):
        return copy.deepcopy(config["model"])
    return None


def checkpoint_blob(state_dict: dict[str, torch.Tensor], model_config: dict[str, Any] | None = None) -> dict[str, Any]:
    blob: dict[str, Any] = {"state_dict": state_dict}
    if model_config:
        blob["model_config"] = copy.deepcopy(model_config)
    return blob


def load_state_dict_file(path: str | Path, map_location: str = "cpu") -> dict[str, torch.Tensor]:
    blob = torch.load(Path(path), map_location=map_location)
    return extract_state_dict(blob)


def load_model_config_file(
    path: str | Path,
    map_location: str = "cpu",
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blob = torch.load(Path(path), map_location=map_location)
    return extract_model_config(blob) or copy.deepcopy(fallback or {"kind": "atennuate"})


def save_checkpoint_file(path: str | Path, state_dict: dict[str, torch.Tensor], model_config: dict[str, Any] | None = None) -> None:
    torch.save(checkpoint_blob(state_dict, model_config=model_config), Path(path))


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
        }

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, value in model.state_dict().items():
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {name: tensor.detach().clone() for name, tensor in state["shadow"].items()}

    def to(self, device: str | torch.device) -> None:
        self.shadow = {name: tensor.to(device) for name, tensor in self.shadow.items()}

    def copy_to_model(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


@dataclass
class TopKCheckpointManager:
    run_dir: Path
    keep: int = 3
    entries: list[dict[str, Any]] = field(default_factory=list)

    def maybe_save(
        self,
        state_dict: dict[str, torch.Tensor],
        metrics: dict[str, float],
        epoch: int,
        tag: str = "val",
        model_config: dict[str, Any] | None = None,
    ) -> bool:
        candidate = {"epoch": int(epoch), "metrics": dict(metrics)}
        should_save = len(self.entries) < self.keep
        if not should_save:
            worst = self.entries[-1]
            should_save = compare_metric_dicts(candidate["metrics"], worst["metrics"])
        if not should_save:
            return False

        ckpt_path = self.run_dir / f"topk_{tag}_epoch{epoch:03d}.pt"
        save_checkpoint_file(ckpt_path, state_dict, model_config=model_config)
        candidate["path"] = ckpt_path.as_posix()
        self.entries.append(candidate)
        self.entries.sort(key=lambda item: (item["metrics"].get("PESQ", float("-inf")), item["metrics"].get("STOI", float("-inf")), item["metrics"].get("SI_SDR", float("-inf"))), reverse=True)
        while len(self.entries) > self.keep:
            stale = self.entries.pop(-1)
            stale_path = Path(stale["path"])
            if stale_path.exists():
                stale_path.unlink()
        return True

    def state_dict(self) -> dict[str, Any]:
        return {"keep": self.keep, "entries": copy.deepcopy(self.entries)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.keep = int(state["keep"])
        self.entries = copy.deepcopy(state["entries"])
