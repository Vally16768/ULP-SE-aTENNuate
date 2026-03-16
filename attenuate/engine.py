from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from attenuate.audio import apply_input_frontend, set_random_seed
from attenuate.checkpoints import (
    ExponentialMovingAverage,
    TopKCheckpointManager,
    load_state_dict_file,
    save_checkpoint_file,
)
from attenuate.data import NoisyAugmentor, VoiceBankDemandDataset
from attenuate.eval_runtime import compare_metric_dicts, evaluate_model_on_manifest
from attenuate.losses import FrozenPESQSurrogateLoss, PerceptualObjective
from attenuate.model import build_model, resolve_model_config


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr: float,
    base_lr: float,
):
    warmup_steps = int(round(total_steps * warmup_ratio))
    min_lr_ratio = min(1.0, float(min_lr) / float(base_lr))

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        step = min(step + 1, total_steps)
        if warmup_steps > 0 and step <= warmup_steps:
            return max(1e-8, step / warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _build_scheduler_bundle(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr: float,
    base_lr: float,
    scheduler_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = dict(scheduler_cfg or {})
    kind = str(cfg.get("kind", "cosine")).strip().lower()
    if kind == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 4)),
            threshold=float(cfg.get("threshold", 0.001)),
            threshold_mode=str(cfg.get("threshold_mode", "rel")),
            cooldown=int(cfg.get("cooldown", 0)),
            min_lr=float(min_lr),
        )
        return {
            "kind": "plateau",
            "scheduler": scheduler,
            "monitor": str(cfg.get("monitor", "PESQ")),
        }

    scheduler = _build_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
        min_lr=min_lr,
        base_lr=base_lr,
    )
    return {
        "kind": "batch",
        "scheduler": scheduler,
        "monitor": str(cfg.get("monitor", "PESQ")),
    }


def _load_surrogate(spec: dict[str, Any] | None, device: str) -> tuple[FrozenPESQSurrogateLoss | None, float, int]:
    if not spec or not spec.get("enabled", False):
        return None, 0.0, 0
    checkpoint = spec.get("checkpoint")
    if not checkpoint:
        return None, 0.0, 0
    from attenuate.pesq_surrogate import PESQSurrogateNet, load_surrogate_checkpoint

    surrogate = PESQSurrogateNet()
    load_surrogate_checkpoint(surrogate, checkpoint, map_location="cpu")
    surrogate.to(device)
    loss = FrozenPESQSurrogateLoss(surrogate)
    return loss, float(spec.get("weight", 0.05)), int(spec.get("warmup_epochs", 3))


def _selection_metrics(
    quick_eval: dict[str, Any] | None,
    full_eval: dict[str, Any] | None,
    fallback_train_loss: float,
) -> dict[str, float]:
    if full_eval is not None:
        return {
            "PESQ": float(full_eval["aggregate"]["PESQ"]),
            "STOI": float(full_eval["aggregate"]["STOI"]),
            "SI_SDR": float(full_eval["aggregate"]["SI_SDR"]),
        }
    if quick_eval is not None:
        return {
            "PESQ": float(quick_eval["aggregate"]["PESQ"]),
            "STOI": float(quick_eval["aggregate"]["STOI"]),
            "SI_SDR": float(quick_eval["aggregate"]["SI_SDR"]),
        }
    return {"PESQ": -float(fallback_train_loss), "STOI": 0.0, "SI_SDR": 0.0}


def _evaluate_objective_on_manifest(
    *,
    model: nn.Module,
    objective: PerceptualObjective,
    manifest_csv: str | Path,
    device: str,
    sample_rate: int,
    segment_len: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_files: int | None,
    frontend_cfg: dict[str, Any] | None,
) -> dict[str, float]:
    dataset = VoiceBankDemandDataset(
        manifest_csv,
        segment_len=segment_len,
        sample_rate=sample_rate,
        augmentor=None,
        seed=seed,
    )
    if max_files is not None:
        dataset.records = dataset.records[: int(max_files)]
    dataset.set_epoch(0, 1)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(device == "cuda"),
        persistent_workers=False,
    )
    component_sums: dict[str, float] = {}
    total = 0.0
    seen = 0
    model.eval()
    objective.eval()
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device).unsqueeze(1)
            clean = clean.to(device).unsqueeze(1)
            noisy_input = apply_input_frontend(noisy.squeeze(1), frontend_cfg, sample_rate=sample_rate).unsqueeze(1)
            enhanced = model(noisy_input)
            loss, components = objective(enhanced, clean, noisy)
            batch_items = int(noisy.shape[0])
            total += float(loss.detach().item()) * batch_items
            seen += batch_items
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + float(value) * batch_items
    return {
        "loss": total / max(1, seen),
        **{f"loss_{key}": value / max(1, seen) for key, value in component_sums.items()},
        "count": float(seen),
    }


def _plot_history(history: list[dict[str, Any]], run_dir: Path) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    if not history:
        return {}

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train"]["loss"]) for item in history]
    val_loss = [
        float(item.get("full_val_loss", {}).get("loss", float("nan")))
        if "full_val_loss" in item
        else float("nan")
        for item in history
    ]
    quick_pesq = [
        float(item.get("quick_val", {}).get("PESQ", float("nan")))
        if "quick_val" in item
        else float("nan")
        for item in history
    ]
    full_pesq = [
        float(item.get("full_val", {}).get("PESQ", float("nan")))
        if "full_val" in item
        else float("nan")
        for item in history
    ]
    full_stoi = [
        float(item.get("full_val", {}).get("STOI", float("nan")))
        if "full_val" in item
        else float("nan")
        for item in history
    ]
    full_sisdr = [
        float(item.get("full_val", {}).get("SI_SDR", float("nan")))
        if "full_val" in item
        else float("nan")
        for item in history
    ]

    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(epochs, train_loss, label="train_loss", linewidth=1.8)
    if any(not math.isnan(value) for value in val_loss):
        ax.plot(epochs, val_loss, label="val_loss_proxy", linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    loss_path = run_dir / "loss_curves.png"
    fig.tight_layout()
    fig.savefig(loss_path.as_posix(), dpi=150)
    plt.close(fig)
    paths["loss_curves"] = loss_path.as_posix()

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.0), sharex=True)
    axes[0].plot(epochs, quick_pesq, label="quick_val_pesq", linewidth=1.5)
    axes[0].plot(epochs, full_pesq, label="full_val_pesq", linewidth=1.8)
    axes[0].set_ylabel("PESQ")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, full_stoi, label="full_val_stoi", linewidth=1.8, color="tab:orange")
    axes[1].set_ylabel("STOI")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[2].plot(epochs, full_sisdr, label="full_val_sisdr", linewidth=1.8, color="tab:green")
    axes[2].set_ylabel("SI-SDR")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    metrics_path = run_dir / "metric_curves.png"
    fig.tight_layout()
    fig.savefig(metrics_path.as_posix(), dpi=150)
    plt.close(fig)
    paths["metric_curves"] = metrics_path.as_posix()

    return paths


def _history_record_for_epoch(history: list[dict[str, Any]], epoch: int) -> dict[str, Any] | None:
    for record in history:
        if int(record["epoch"]) == int(epoch):
            return record
    return None


def _latest_eval_record(history: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for record in reversed(history):
        if key in record:
            return record[key]
    return None


def _diagnose_history(
    *,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_eval_kind: str,
    total_epochs_config: int,
    early_stop_patience: int,
    final_epoch: int,
) -> dict[str, Any]:
    best_record = _history_record_for_epoch(history, best_epoch) if best_epoch > 0 else None
    last_record = history[-1] if history else None
    full_pesq = [float(item["full_val"]["PESQ"]) for item in history if "full_val" in item]
    oscillation = 0.0
    if len(full_pesq) >= 3:
        diffs = [abs(b - a) for a, b in zip(full_pesq[:-1], full_pesq[1:])]
        oscillation = float(sum(diffs) / len(diffs))

    latest_full = _latest_eval_record(history, "full_val")
    latest_quick = _latest_eval_record(history, "quick_val")
    best_metrics = None
    if best_record is not None:
        if "full_val" in best_record:
            best_metrics = best_record["full_val"]
        elif "quick_val" in best_record:
            best_metrics = best_record["quick_val"]
        else:
            best_metrics = best_record["selection"]

    best_train_loss = float(best_record["train"]["loss"]) if best_record is not None else None
    best_val_loss = None
    if best_record is not None:
        if "full_val_loss" in best_record:
            best_val_loss = float(best_record["full_val_loss"]["loss"])
        elif "quick_val_loss" in best_record:
            best_val_loss = float(best_record["quick_val_loss"]["loss"])

    clip_detected = bool(
        any(float(item.get("full_val", {}).get("clip_fraction", 0.0)) > 1e-4 for item in history)
        or any(float(item.get("quick_val", {}).get("clip_fraction", 0.0)) > 1e-4 for item in history)
    )
    non_finite_detected = bool(
        any(bool(item.get("full_val", {}).get("has_non_finite", False)) for item in history)
        or any(bool(item.get("quick_val", {}).get("has_non_finite", False)) for item in history)
    )
    over_attenuation_signal = bool(
        latest_full is not None
        and float(latest_full.get("DELTA_SNR", 0.0)) > 3.0
        and float(latest_full.get("STOI", 1.0)) < 0.90
    )
    early_plateau = bool(best_epoch > 0 and best_epoch <= max(3, total_epochs_config // 5))
    val_gap = None if best_train_loss is None or best_val_loss is None else best_val_loss - best_train_loss

    improvable = bool(
        non_finite_detected
        or clip_detected
        or oscillation > 0.03
        or over_attenuation_signal
        or (val_gap is not None and val_gap > 0.15)
        or early_plateau
    )

    return {
        "best_epoch": int(best_epoch),
        "best_eval_kind": best_eval_kind,
        "final_epoch": int(final_epoch),
        "stop_reason": "early_stop" if final_epoch < total_epochs_config else "max_epochs",
        "early_stop_patience": int(early_stop_patience),
        "best_metrics": best_metrics,
        "last_metrics": latest_full or latest_quick or (last_record["selection"] if last_record else None),
        "train_vs_val_loss_gap": val_gap,
        "signals": {
            "selection_oscillation": oscillation,
            "clip_detected": clip_detected,
            "non_finite_detected": non_finite_detected,
            "over_attenuation_suspected": over_attenuation_signal,
            "early_plateau": early_plateau,
        },
        "verdict": "improvable" if improvable else "control",
    }


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    train_csv = cfg["train_csv"]
    sample_rate = int(cfg.get("sample_rate", 16000))
    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 1337))
    set_random_seed(seed)

    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.get("run_dir", "runs/default"))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = Path(cfg.get("checkpoint_out", run_dir / "best.pt"))
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    last_state_path = run_dir / "last_train_state.pt"
    metrics_jsonl = run_dir / "metrics.jsonl"
    summary_json = run_dir / "summary.json"
    diagnostics_json = run_dir / "diagnostics.json"
    if not cfg.get("resume") and metrics_jsonl.exists():
        metrics_jsonl.unlink()

    model_cfg = resolve_model_config(cfg.get("model"))
    augmentor = NoisyAugmentor(sample_rate=sample_rate, config=cfg.get("augment", {}))
    dataset = VoiceBankDemandDataset(
        train_csv,
        segment_len=int(cfg.get("segment_len", 32000)),
        sample_rate=sample_rate,
        augmentor=augmentor,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(device == "cuda"),
        persistent_workers=bool(num_workers > 0),
    )
    if len(loader) == 0:
        raise ValueError("Training DataLoader has zero batches")

    model = build_model(model_cfg).to(device)
    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint:
        model.load_state_dict(load_state_dict_file(init_checkpoint, map_location="cpu"))

    objective_cfg = dict(cfg.get("objective", {}))
    frontend_cfg = dict(cfg.get("frontend", {}))
    objective = PerceptualObjective(
        sample_rate=sample_rate,
        wave_beta=float(objective_cfg.get("wave_beta", 0.5)),
        erb_weight=float(objective_cfg.get("erb_weight", 0.0)),
        mrstft_weight=float(objective_cfg.get("mrstft_weight", 0.0)),
        complex_weight=float(objective_cfg.get("complex_weight", 0.0)),
        sisdr_weight=float(objective_cfg.get("sisdr_weight", 0.0)),
        high_snr_weight=float(objective_cfg.get("high_snr_weight", 0.0)),
        band_emphasis_strength=float(objective_cfg.get("band_emphasis_strength", 0.0)),
        high_snr_threshold_db=float(objective_cfg.get("high_snr_threshold_db", 15.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.02)),
    )
    scheduler_bundle = _build_scheduler_bundle(
        optimizer=optimizer,
        total_steps=epochs * len(loader),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        min_lr=float(cfg.get("min_lr", 1e-6)),
        base_lr=float(cfg.get("lr", 1e-3)),
        scheduler_cfg=cfg.get("scheduler"),
    )
    scheduler = scheduler_bundle["scheduler"]
    scheduler_kind = str(scheduler_bundle["kind"])
    scheduler_monitor = str(scheduler_bundle.get("monitor", "PESQ"))

    ema_decay = float(cfg.get("ema_decay", 0.0))
    ema = ExponentialMovingAverage(model, ema_decay) if ema_decay > 0 else None
    topk = TopKCheckpointManager(run_dir=run_dir, keep=int(cfg.get("save_top_k", 3)))

    surrogate_loss, surrogate_weight, surrogate_warmup_epochs = _load_surrogate(cfg.get("stage2"), device)

    best_selection: dict[str, float] | None = None
    best_epoch = 0
    best_eval_kind = "train"
    epochs_since_improve = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    global_step = 0

    resume_path = cfg.get("resume")
    if resume_path:
        resume_blob = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(resume_blob["model_state"])
        optimizer.load_state_dict(resume_blob["optimizer_state"])
        scheduler.load_state_dict(resume_blob["scheduler_state"])
        if ema is not None and resume_blob.get("ema_state") is not None:
            ema.load_state_dict(resume_blob["ema_state"])
            ema.to(device)
        if resume_blob.get("topk_state") is not None:
            topk.load_state_dict(resume_blob["topk_state"])
        best_selection = resume_blob.get("best_selection")
        best_epoch = int(resume_blob.get("best_epoch", 0))
        best_eval_kind = str(resume_blob.get("best_eval_kind", "train"))
        epochs_since_improve = int(resume_blob.get("epochs_since_improve", 0))
        history = list(resume_blob.get("history", []))
        start_epoch = int(resume_blob["epoch"]) + 1
        global_step = int(resume_blob.get("global_step", 0))

    eval_model = build_model(model_cfg).to(device)

    def current_eval_state(*, cpu: bool = False) -> dict[str, torch.Tensor]:
        if ema is not None:
            source = ema.shadow
        else:
            source = model.state_dict()
        if cpu:
            return {key: value.detach().cpu().clone() for key, value in source.items()}
        return {key: value.detach().clone() for key, value in source.items()}

    def run_eval(manifest_key: str) -> dict[str, Any] | None:
        manifest = cfg.get(manifest_key)
        if not manifest:
            return None
        eval_model.load_state_dict(current_eval_state(), strict=True)
        result = evaluate_model_on_manifest(
            eval_model,
            manifest_csv=manifest,
            device=device,
            sample_rate=sample_rate,
            max_files=cfg.get("max_eval_files"),
            desc=manifest_key,
            frontend_cfg=frontend_cfg,
        )
        result["objective"] = _evaluate_objective_on_manifest(
            model=eval_model,
            objective=objective,
            manifest_csv=manifest,
            device=device,
            sample_rate=sample_rate,
            segment_len=int(cfg.get("segment_len", 32000)),
            batch_size=max(1, min(batch_size, 4)),
            num_workers=num_workers,
            seed=seed,
            max_files=cfg.get("diagnostics_loss_files", cfg.get("max_eval_files", 64)),
            frontend_cfg=frontend_cfg,
        )
        return result

    early_stop_patience = int(cfg.get("early_stop_patience", 6))
    grad_clip = float(cfg.get("grad_clip", 0.0))
    eval_every = int(cfg.get("eval_every", 2))
    full_val_every = int(cfg.get("full_val_every", 5))
    full_val_on_quick_best = bool(cfg.get("full_val_on_quick_best", True))

    for epoch in range(start_epoch, epochs + 1):
        dataset.set_epoch(epoch, epochs)
        model.train()
        running_total = 0.0
        component_sums: dict[str, float] = {}
        seen = 0

        progress = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", unit="batch")
        for noisy, clean in progress:
            noisy = noisy.to(device).unsqueeze(1)
            clean = clean.to(device).unsqueeze(1)
            noisy_input = apply_input_frontend(noisy.squeeze(1), frontend_cfg, sample_rate=sample_rate).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            enhanced = model(noisy_input)
            total_loss, components = objective(enhanced, clean, noisy)

            if surrogate_loss is not None and epoch > surrogate_warmup_epochs:
                stage_weight = surrogate_weight * min(1.0, (epoch - surrogate_warmup_epochs) / max(1, epochs - surrogate_warmup_epochs))
                pesq_aux = surrogate_loss(enhanced, clean)
                total_loss = total_loss + stage_weight * pesq_aux
                components["surrogate"] = float(pesq_aux.detach().item())
                components["surrogate_weight"] = stage_weight

            total_loss.backward()
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler_kind == "batch":
                scheduler.step()
            if ema is not None:
                ema.update(model)

            batch_size_now = int(noisy.shape[0])
            running_total += float(total_loss.detach().item()) * batch_size_now
            seen += batch_size_now
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + float(value) * batch_size_now
            global_step += 1

            mean_total = running_total / max(1, seen)
            progress.set_postfix({"loss": f"{mean_total:.5f}", "lr": f"{optimizer.param_groups[0]['lr']:.3e}"})

        train_metrics = {
            "loss": running_total / max(1, seen),
            **{f"loss_{key}": value / max(1, seen) for key, value in component_sums.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        quick_eval = None
        full_eval = None
        new_quick_best = False
        if cfg.get("quick_val_csv") and epoch % max(1, eval_every) == 0:
            quick_eval = run_eval("quick_val_csv")
            current_quick = {
                "PESQ": float(quick_eval["aggregate"]["PESQ"]),
                "STOI": float(quick_eval["aggregate"]["STOI"]),
                "SI_SDR": float(quick_eval["aggregate"]["SI_SDR"]),
            }
            new_quick_best = compare_metric_dicts(current_quick, best_selection)

        should_full_eval = False
        if cfg.get("val_csv"):
            if epoch % max(1, full_val_every) == 0:
                should_full_eval = True
            if full_val_on_quick_best and new_quick_best:
                should_full_eval = True
            if epoch == epochs:
                should_full_eval = True

        if should_full_eval:
            full_eval = run_eval("val_csv")

        if scheduler_kind == "plateau" and full_eval is not None:
            scheduler_value = float(full_eval["aggregate"].get(scheduler_monitor, float("nan")))
            if math.isfinite(scheduler_value):
                scheduler.step(scheduler_value)

        selection = _selection_metrics(quick_eval, full_eval, fallback_train_loss=train_metrics["loss"])
        improved = compare_metric_dicts(selection, best_selection)
        if improved:
            best_selection = dict(selection)
            best_epoch = epoch
            best_eval_kind = "full_val" if full_eval is not None else ("quick_val" if quick_eval is not None else "train")
            epochs_since_improve = 0
            best_state = current_eval_state(cpu=True)
            save_checkpoint_file(checkpoint_out, best_state, model_config=model_cfg)
            save_checkpoint_file(run_dir / "best.pt", best_state, model_config=model_cfg)
            topk.maybe_save(best_state, metrics=selection, epoch=epoch, tag=best_eval_kind, model_config=model_cfg)
        elif quick_eval is not None or full_eval is not None:
            epochs_since_improve += 1

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train": train_metrics,
            "selection": selection,
            "best_selection": best_selection,
            "improved": improved,
            "lr_after_scheduler": float(optimizer.param_groups[0]["lr"]),
        }
        if quick_eval is not None:
            epoch_record["quick_val"] = quick_eval["aggregate"]
            epoch_record["quick_val_loss"] = quick_eval["objective"]
        if full_eval is not None:
            epoch_record["full_val"] = full_eval["aggregate"]
            epoch_record["full_val_loss"] = full_eval["objective"]
        history.append(epoch_record)

        with metrics_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "config": cfg,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "ema_state": ema.state_dict() if ema is not None else None,
                "topk_state": topk.state_dict(),
                "best_selection": best_selection,
                "best_epoch": best_epoch,
                "best_eval_kind": best_eval_kind,
                "epochs_since_improve": epochs_since_improve,
                "history": history,
            },
            last_state_path,
        )

        if epochs_since_improve >= early_stop_patience:
            break

    plot_paths = _plot_history(history, run_dir)
    diagnostics = _diagnose_history(
        history=history,
        best_epoch=best_epoch,
        best_eval_kind=best_eval_kind,
        total_epochs_config=epochs,
        early_stop_patience=early_stop_patience,
        final_epoch=int(history[-1]["epoch"]) if history else 0,
    )
    diagnostics_json.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    summary = {
        "config": cfg,
        "device": device,
        "best_epoch": best_epoch,
        "best_eval_kind": best_eval_kind,
        "best_selection": best_selection,
        "checkpoint_out": checkpoint_out.as_posix(),
        "best_checkpoint": (run_dir / "best.pt").as_posix(),
        "last_state": last_state_path.as_posix(),
        "metrics_jsonl": metrics_jsonl.as_posix(),
        "diagnostics_json": diagnostics_json.as_posix(),
        "plot_paths": plot_paths,
        "diagnostics": diagnostics,
        "history_length": len(history),
        "final_epoch": int(history[-1]["epoch"]) if history else 0,
        "topk": topk.entries,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
