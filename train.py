from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

from attenuate.engine import run_training


def _load_config_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train aTENNuate with PESQ-first experiment support.")
    parser.add_argument("--config", type=str, default=None, help="Optional TOML/JSON config.")
    parser.add_argument("--train-csv", type=str, default=None, help="CSV with columns noisy,clean.")
    parser.add_argument("--val-csv", type=str, default=None, help="Validation CSV with columns noisy,clean.")
    parser.add_argument("--quick-val-csv", type=str, default=None, help="Fast validation CSV with columns noisy,clean.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory.")
    parser.add_argument("--checkpoint-out", type=str, default=None, help="Best checkpoint output path.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from last_train_state.pt.")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Initialize weights from checkpoint.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--segment-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--full-val-every", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--save-top-k", type=int, default=None)
    parser.add_argument("--max-eval-files", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=None, help="Training/evaluation sample rate.")
    parser.add_argument("--model-kind", type=str, default=None, help="Model backend: atennuate, mp_senet_lite, mp_senet_micro or percepnet_class.")
    parser.add_argument("--scheduler-kind", type=str, default=None, help="Scheduler kind: cosine or reduce_on_plateau.")
    parser.add_argument("--scheduler-factor", type=float, default=None, help="LR reduction factor for adaptive scheduler.")
    parser.add_argument("--scheduler-patience", type=int, default=None, help="Patience for adaptive LR scheduler.")
    parser.add_argument("--scheduler-threshold", type=float, default=None, help="Minimum monitored improvement for adaptive scheduler.")
    parser.add_argument("--scheduler-cooldown", type=int, default=None, help="Cooldown epochs for adaptive scheduler.")

    parser.add_argument("--wave-beta", type=float, default=None)
    parser.add_argument("--erb-weight", type=float, default=None)
    parser.add_argument("--mrstft-weight", type=float, default=None)
    parser.add_argument("--complex-weight", type=float, default=None)
    parser.add_argument("--sisdr-weight", type=float, default=None)
    parser.add_argument("--high-snr-weight", type=float, default=None)
    parser.add_argument("--high-snr-threshold-db", type=float, default=None)
    parser.add_argument("--band-emphasis-strength", type=float, default=None)

    parser.add_argument("--augment", action="store_true", help="Enable training augmentations.")
    parser.add_argument("--augment-probability", type=float, default=None)
    parser.add_argument("--augment-max-ops", type=int, default=None)
    parser.add_argument("--augment-ramp-ratio", type=float, default=None)

    parser.add_argument("--spectral-gate", action="store_true", help="Apply spectral gating before the model.")
    parser.add_argument("--gate-noise-quantile", type=float, default=None)
    parser.add_argument("--gate-threshold-scale", type=float, default=None)
    parser.add_argument("--gate-mask-slope", type=float, default=None)
    parser.add_argument("--gate-mask-floor", type=float, default=None)

    parser.add_argument("--stage2-surrogate-checkpoint", type=str, default=None)
    parser.add_argument("--stage2-surrogate-weight", type=float, default=None)
    parser.add_argument("--stage2-surrogate-warmup-epochs", type=int, default=None)
    return parser.parse_args()


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _cli_updates(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "model": {},
        "objective": {},
        "augment": {},
        "frontend": {},
        "stage2": {},
        "scheduler": {},
    }

    top_level = {
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "quick_val_csv": args.quick_val_csv,
        "run_dir": args.run_dir,
        "checkpoint_out": args.checkpoint_out,
        "resume": args.resume,
        "init_checkpoint": args.init_checkpoint,
        "seed": args.seed,
        "device": args.device,
        "sample_rate": args.sample_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "segment_len": args.segment_len,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "ema_decay": args.ema_decay,
        "eval_every": args.eval_every,
        "full_val_every": args.full_val_every,
        "early_stop_patience": args.early_stop_patience,
        "save_top_k": args.save_top_k,
        "max_eval_files": args.max_eval_files,
    }
    for key, value in top_level.items():
        if value is not None:
            cfg[key] = value

    if args.model_kind is not None:
        cfg["model"]["kind"] = args.model_kind

    scheduler = {
        "kind": args.scheduler_kind,
        "factor": args.scheduler_factor,
        "patience": args.scheduler_patience,
        "threshold": args.scheduler_threshold,
        "cooldown": args.scheduler_cooldown,
    }
    for key, value in scheduler.items():
        if value is not None:
            cfg["scheduler"][key] = value

    objective = {
        "wave_beta": args.wave_beta,
        "erb_weight": args.erb_weight,
        "mrstft_weight": args.mrstft_weight,
        "complex_weight": args.complex_weight,
        "sisdr_weight": args.sisdr_weight,
        "high_snr_weight": args.high_snr_weight,
        "high_snr_threshold_db": args.high_snr_threshold_db,
        "band_emphasis_strength": args.band_emphasis_strength,
    }
    for key, value in objective.items():
        if value is not None:
            cfg["objective"][key] = value

    if args.augment:
        cfg["augment"]["enabled"] = True
    augment = {
        "probability": args.augment_probability,
        "max_ops": args.augment_max_ops,
        "ramp_ratio": args.augment_ramp_ratio,
    }
    for key, value in augment.items():
        if value is not None:
            cfg["augment"][key] = value

    if args.spectral_gate:
        cfg["frontend"]["kind"] = "spectral_gate"
    frontend = {
        "noise_quantile": args.gate_noise_quantile,
        "threshold_scale": args.gate_threshold_scale,
        "mask_slope": args.gate_mask_slope,
        "mask_floor": args.gate_mask_floor,
    }
    for key, value in frontend.items():
        if value is not None:
            cfg["frontend"][key] = value

    if args.stage2_surrogate_checkpoint is not None:
        cfg["stage2"]["enabled"] = True
        cfg["stage2"]["checkpoint"] = args.stage2_surrogate_checkpoint
    if args.stage2_surrogate_weight is not None:
        cfg["stage2"]["weight"] = args.stage2_surrogate_weight
    if args.stage2_surrogate_warmup_epochs is not None:
        cfg["stage2"]["warmup_epochs"] = args.stage2_surrogate_warmup_epochs

    return cfg


def _default_config() -> dict[str, Any]:
    return {
        "epochs": 80,
        "batch_size": 4,
        "num_workers": 0,
        "segment_len": 64000,
        "sample_rate": 16000,
        "lr": 1e-3,
        "min_lr": 1e-6,
        "warmup_ratio": 0.05,
        "weight_decay": 0.02,
        "grad_clip": 5.0,
        "ema_decay": 0.999,
        "eval_every": 2,
        "full_val_every": 5,
        "full_val_on_quick_best": True,
        "early_stop_patience": 8,
        "save_top_k": 3,
        "model": {
            "kind": "atennuate",
        },
        "scheduler": {
            "kind": "cosine",
            "factor": 0.5,
            "patience": 4,
            "threshold": 0.001,
            "cooldown": 0,
        },
        "objective": {
            "wave_beta": 0.5,
            "erb_weight": 1.0,
            "mrstft_weight": 0.0,
            "complex_weight": 0.0,
            "sisdr_weight": 0.0,
            "high_snr_weight": 0.0,
            "high_snr_threshold_db": 15.0,
            "band_emphasis_strength": 0.0,
        },
        "augment": {
            "enabled": False,
            "probability": 0.75,
            "max_ops": 2,
            "ramp_ratio": 0.3,
        },
        "frontend": {
            "kind": "none",
        },
        "stage2": {
            "enabled": False,
        },
    }


def main() -> None:
    args = _parse_args()
    config = _default_config()
    config = _deep_update(config, _load_config_file(args.config))
    config = _deep_update(config, _cli_updates(args))

    if not config.get("train_csv"):
        raise ValueError("--train-csv is required unless provided by --config")

    run_training(config)


if __name__ == "__main__":
    main()
