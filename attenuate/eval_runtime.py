from __future__ import annotations

import csv
import math
import statistics as st
from pathlib import Path
from typing import Any

import numpy as np
import torch

from attenuate.audio import apply_input_frontend, load_mono_audio
from attenuate.data import PairRecord, load_manifest_records
from metrics.pesq import pesq_score
from metrics.sisdr import sisdr
from metrics.snr import delta_snr
from metrics.stoi import stoi_score


def compare_metric_dicts(candidate: dict[str, float] | None, incumbent: dict[str, float] | None) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    key_order = ("PESQ", "STOI", "SI_SDR")
    for key in key_order:
        cand = float(candidate.get(key, float("-inf")))
        base = float(incumbent.get(key, float("-inf")))
        if cand > base + 1e-8:
            return True
        if cand < base - 1e-8:
            return False
    return False


def clipping_fraction(x: np.ndarray, threshold: float = 0.999) -> float:
    return float(np.mean(np.abs(x) >= threshold))


def compute_intrusive_metrics(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray, sample_rate: int) -> dict[str, float]:
    n = min(len(clean), len(noisy), len(enhanced))
    if n <= 0:
        raise ValueError("Empty audio while computing metrics")
    clean = clean[:n].astype(np.float32)
    noisy = noisy[:n].astype(np.float32)
    enhanced = enhanced[:n].astype(np.float32)
    return {
        "PESQ": float(pesq_score(clean, enhanced, sample_rate)),
        "STOI": float(stoi_score(clean, enhanced, sample_rate, extended=False)),
        "DELTA_SNR": float(delta_snr(clean, noisy, enhanced)),
        "SI_SDR": float(sisdr(clean, enhanced)),
    }


def _rows_for_manifest(manifest_csv: str | Path, max_files: int | None = None) -> list[PairRecord]:
    rows = load_manifest_records(manifest_csv)
    return rows if max_files is None else rows[: int(max_files)]


def evaluate_model_on_manifest(
    model: torch.nn.Module,
    manifest_csv: str | Path,
    device: str,
    sample_rate: int = 16000,
    max_files: int | None = None,
    desc: str | None = None,
    frontend_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = _rows_for_manifest(manifest_csv, max_files=max_files)
    per_file: list[dict[str, Any]] = []
    model.eval()

    for record in rows:
        clean, _ = load_mono_audio(record.clean, sample_rate)
        noisy, _ = load_mono_audio(record.noisy, sample_rate)
        with torch.no_grad():
            enhanced = model.denoise_single(
                noisy.unsqueeze(0).to(device),
                frontend_cfg=frontend_cfg,
                sample_rate=sample_rate,
            ).squeeze(0).cpu()

        clean_np = clean.numpy()
        noisy_np = noisy.numpy()
        enhanced_np = enhanced.numpy()
        metrics = compute_intrusive_metrics(clean_np, noisy_np, enhanced_np, sample_rate)
        metrics["clip_fraction"] = clipping_fraction(enhanced_np)
        metrics["finite"] = float(np.isfinite(enhanced_np).all())
        per_file.append(
            {
                "utterance_id": record.utterance_id,
                "speaker_id": record.speaker_id,
                "clean": record.clean.as_posix(),
                "noisy": record.noisy.as_posix(),
                **metrics,
            }
        )

    aggregate = {
        key: float(st.mean(float(row[key]) for row in per_file))
        for key in ("PESQ", "STOI", "DELTA_SNR", "SI_SDR", "clip_fraction", "finite")
    }
    aggregate["count"] = len(per_file)
    aggregate["desc"] = desc or Path(manifest_csv).stem
    aggregate["has_non_finite"] = bool(any(not math.isfinite(float(row["PESQ"])) for row in per_file) or aggregate["finite"] < 1.0)
    return {"aggregate": aggregate, "rows": per_file}


def evaluate_frontend_on_manifest(
    manifest_csv: str | Path,
    frontend_cfg: dict[str, Any],
    sample_rate: int = 16000,
    max_files: int | None = None,
    desc: str | None = None,
) -> dict[str, Any]:
    rows = _rows_for_manifest(manifest_csv, max_files=max_files)
    per_file: list[dict[str, Any]] = []
    for record in rows:
        clean, _ = load_mono_audio(record.clean, sample_rate)
        noisy, _ = load_mono_audio(record.noisy, sample_rate)
        enhanced = apply_input_frontend(noisy.unsqueeze(0), frontend_cfg, sample_rate=sample_rate).squeeze(0)
        metrics = compute_intrusive_metrics(clean.numpy(), noisy.numpy(), enhanced.numpy(), sample_rate)
        metrics["clip_fraction"] = clipping_fraction(enhanced.numpy())
        metrics["finite"] = float(np.isfinite(enhanced.numpy()).all())
        per_file.append(
            {
                "utterance_id": record.utterance_id,
                "speaker_id": record.speaker_id,
                "clean": record.clean.as_posix(),
                "noisy": record.noisy.as_posix(),
                **metrics,
            }
        )

    aggregate = {
        key: float(st.mean(float(row[key]) for row in per_file))
        for key in ("PESQ", "STOI", "DELTA_SNR", "SI_SDR", "clip_fraction", "finite")
    }
    aggregate["count"] = len(per_file)
    aggregate["desc"] = desc or "frontend_only"
    aggregate["has_non_finite"] = bool(any(not math.isfinite(float(row["PESQ"])) for row in per_file) or aggregate["finite"] < 1.0)
    return {"aggregate": aggregate, "rows": per_file}


def evaluate_noisy_baseline(
    manifest_csv: str | Path,
    sample_rate: int = 16000,
    max_files: int | None = None,
) -> dict[str, Any]:
    rows = _rows_for_manifest(manifest_csv, max_files=max_files)
    per_file: list[dict[str, Any]] = []
    for record in rows:
        clean, _ = load_mono_audio(record.clean, sample_rate)
        noisy, _ = load_mono_audio(record.noisy, sample_rate)
        metrics = compute_intrusive_metrics(clean.numpy(), noisy.numpy(), noisy.numpy(), sample_rate)
        metrics["clip_fraction"] = clipping_fraction(noisy.numpy())
        metrics["finite"] = 1.0
        per_file.append({"utterance_id": record.utterance_id, "speaker_id": record.speaker_id, **metrics})
    aggregate = {
        key: float(st.mean(float(row[key]) for row in per_file))
        for key in ("PESQ", "STOI", "DELTA_SNR", "SI_SDR", "clip_fraction", "finite")
    }
    aggregate["count"] = len(per_file)
    aggregate["desc"] = "noisy_baseline"
    aggregate["has_non_finite"] = False
    return {"aggregate": aggregate, "rows": per_file}


def write_rows_csv(rows: list[dict[str, Any]], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
