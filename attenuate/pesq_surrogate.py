from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from attenuate.audio import load_mono_audio, set_random_seed, stack_with_lengths
from attenuate.checkpoints import load_model_config_file, load_state_dict_file
from attenuate.eval_runtime import compute_intrusive_metrics
from attenuate.losses import ERBFeatureProjector
from attenuate.model import build_model


class PESQSurrogateNet(nn.Module):
    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, n_bands: int = 64) -> None:
        super().__init__()
        self.projector = ERBFeatureProjector(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_bands=n_bands,
            log_features=True,
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, clean: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        clean_feat = self.projector(clean)
        enhanced_feat = self.projector(enhanced)
        diff_feat = torch.abs(clean_feat - enhanced_feat)
        x = torch.stack([clean_feat, enhanced_feat, diff_feat], dim=1)
        x = self.encoder(x)
        score = self.head(x).squeeze(-1)
        return 4.5 * torch.sigmoid(score)


def load_surrogate_checkpoint(model: PESQSurrogateNet, path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    blob = torch.load(Path(path), map_location=map_location)
    state_dict = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    model.load_state_dict(state_dict)
    return blob if isinstance(blob, dict) else {"model_state": state_dict}


class SurrogateExampleDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def surrogate_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    clean, _ = stack_with_lengths([item["clean"] for item in batch])
    enhanced, _ = stack_with_lengths([item["enhanced"] for item in batch])
    target = torch.tensor([float(item["target_pesq"]) for item in batch], dtype=torch.float32)
    return {"clean": clean, "enhanced": enhanced, "target": target}


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(x_arr) < 1e-8 or np.std(y_arr) < 1e-8:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def train_surrogate(
    samples_path: str | Path,
    out_dir: str | Path,
    *,
    device: str | None = None,
    seed: int = 1337,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(seed)

    payload = torch.load(Path(samples_path), map_location="cpu")
    samples = list(payload["samples"])
    if len(samples) < 10:
        raise ValueError("Need at least 10 surrogate samples")
    random.Random(seed).shuffle(samples)
    split = max(1, int(len(samples) * (1.0 - val_ratio)))
    train_samples = samples[:split]
    val_samples = samples[split:]
    if not val_samples:
        val_samples = train_samples[-max(1, len(train_samples) // 5) :]
        train_samples = train_samples[: len(train_samples) - len(val_samples)]

    train_loader = DataLoader(SurrogateExampleDataset(train_samples), batch_size=batch_size, shuffle=True, collate_fn=surrogate_collate)
    val_loader = DataLoader(SurrogateExampleDataset(val_samples), batch_size=batch_size, shuffle=False, collate_fn=surrogate_collate)

    model = PESQSurrogateNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.25)
    best_corr = -1.0
    best_path = out_dir / "best_surrogate.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        count = 0
        for batch in train_loader:
            clean = batch["clean"].to(device)
            enhanced = batch["enhanced"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(clean, enhanced)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().item()) * target.shape[0]
            count += int(target.shape[0])

        model.eval()
        val_targets: list[float] = []
        val_preds: list[float] = []
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                clean = batch["clean"].to(device)
                enhanced = batch["enhanced"].to(device)
                target = batch["target"].to(device)
                pred = model(clean, enhanced)
                loss = criterion(pred, target)
                val_loss += float(loss.detach().item()) * target.shape[0]
                val_count += int(target.shape[0])
                val_targets.extend(target.cpu().tolist())
                val_preds.extend(pred.cpu().tolist())

        val_corr = _pearson(val_targets, val_preds)
        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, count),
            "val_loss": val_loss / max(1, val_count),
            "val_corr": val_corr,
        }
        history.append(record)
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "history": history,
                    "best_corr": best_corr,
                    "samples_path": Path(samples_path).as_posix(),
                },
                best_path,
            )

    summary = {
        "best_corr": best_corr,
        "best_checkpoint": best_path.as_posix(),
        "history": history,
        "num_samples": len(samples),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_surrogate_samples(
    manifests: list[str | Path],
    checkpoints: list[str | Path],
    out_path: str | Path,
    *,
    device: str | None = None,
    sample_rate: int = 16000,
    max_rows_per_manifest: int = 96,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from attenuate.data import load_manifest_records

    samples: list[dict[str, Any]] = []
    checkpoints = [Path(path) for path in checkpoints]

    models: list[tuple[str, nn.Module]] = []
    for checkpoint in checkpoints:
        model_cfg = load_model_config_file(checkpoint, fallback={"kind": "atennuate"})
        model = build_model(model_cfg).to(device)
        model.load_state_dict(load_state_dict_file(checkpoint, map_location="cpu"))
        model.eval()
        models.append((checkpoint.stem, model))

    for manifest in manifests:
        records = load_manifest_records(manifest)[:max_rows_per_manifest]
        for record in tqdm(records, desc=f"surrogate-cache:{Path(manifest).stem}", unit="file"):
            clean, _ = load_mono_audio(record.clean, sample_rate)
            noisy, _ = load_mono_audio(record.noisy, sample_rate)

            noisy_metrics = compute_intrusive_metrics(clean.numpy(), noisy.numpy(), noisy.numpy(), sample_rate)
            samples.append(
                {
                    "clean": clean.clone(),
                    "enhanced": noisy.clone(),
                    "target_pesq": float(noisy_metrics["PESQ"]),
                    "source": "noisy_baseline",
                    "utterance_id": record.utterance_id,
                }
            )

            for source_name, model in models:
                with torch.no_grad():
                    enhanced = model.denoise_single(noisy.unsqueeze(0).to(device)).squeeze(0).cpu()
                metrics = compute_intrusive_metrics(clean.numpy(), noisy.numpy(), enhanced.numpy(), sample_rate)
                samples.append(
                    {
                        "clean": clean.clone(),
                        "enhanced": enhanced.clone(),
                        "target_pesq": float(metrics["PESQ"]),
                        "source": source_name,
                        "utterance_id": record.utterance_id,
                    }
                )

    payload = {"samples": samples, "count": len(samples), "checkpoints": [path.as_posix() for path in checkpoints]}
    torch.save(payload, out_path)
    summary = {"count": len(samples), "out_path": out_path.as_posix()}
    (out_path.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
