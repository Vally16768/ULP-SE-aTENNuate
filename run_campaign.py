from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import torch

from download_voicebank_demand import ensure_voicebank_raw_dataset
from attenuate.engine import run_training
from attenuate.eval_runtime import compare_metric_dicts, evaluate_model_on_manifest
from attenuate.locks import ProcessLock
from attenuate.model import build_model
from attenuate.pesq_surrogate import build_surrogate_samples, train_surrogate
from prepare_voicebank_16k import prepare_voicebank_dataset


def _load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_completed_stage_summary(run_dir: str | Path) -> dict[str, Any] | None:
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = _read_json(summary_path)
    except Exception:
        return None
    checkpoint = Path(summary.get("best_checkpoint", run_dir / "best.pt"))
    if not checkpoint.exists():
        return None
    return summary


def _run_or_resume_training(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = Path(cfg["run_dir"])
    existing_summary = _load_completed_stage_summary(run_dir)
    if existing_summary is not None:
        return existing_summary, {"reused": True, "resumed": False}

    resolved_cfg = dict(cfg)
    resume_path = run_dir / "last_train_state.pt"
    resumed = False
    if resolved_cfg.get("resume") is None and resume_path.exists():
        resolved_cfg["resume"] = resume_path.as_posix()
        resumed = True

    summary = run_training(resolved_cfg)
    return summary, {"reused": False, "resumed": resumed}


def _load_model(checkpoint: str | Path, device: str, model_cfg: dict[str, Any] | None = None) -> torch.nn.Module:
    from attenuate.checkpoints import load_model_config_file, load_state_dict_file

    resolved_cfg = load_model_config_file(checkpoint, fallback=model_cfg)
    model = build_model(resolved_cfg).to(device)
    model.load_state_dict(load_state_dict_file(checkpoint, map_location="cpu"))
    model.eval()
    return model


def _evaluate_checkpoint(
    checkpoint: str | Path,
    manifest_csv: str | Path,
    device: str,
    sample_rate: int = 16000,
    max_files: int | None = None,
    frontend_cfg: dict[str, Any] | None = None,
    model_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = _load_model(checkpoint, device, model_cfg=model_cfg)
    return evaluate_model_on_manifest(
        model,
        manifest_csv,
        device=device,
        sample_rate=sample_rate,
        max_files=max_files,
        frontend_cfg=frontend_cfg,
    )


def _write_leaderboard(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sort_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics.get("PESQ", float("-inf"))),
        float(metrics.get("STOI", float("-inf"))),
        float(metrics.get("SI_SDR", float("-inf"))),
    )


def _summary_row(name: str, stage: str, metrics: dict[str, Any], checkpoint: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "experiment": name,
        "stage": stage,
        "checkpoint": checkpoint,
        "PESQ": float(metrics["PESQ"]),
        "STOI": float(metrics["STOI"]),
        "SI_SDR": float(metrics["SI_SDR"]),
        "DELTA_SNR": float(metrics["DELTA_SNR"]),
        "clip_fraction": float(metrics.get("clip_fraction", 0.0)),
    }
    if extra:
        row.update(extra)
    return row


def _gpu_memory_gb(device: str) -> float:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 0.0
    properties = torch.cuda.get_device_properties(torch.device(device))
    return float(properties.total_memory) / float(1024**3)


def _validate_prepared_manifests(paths: dict[str, str], *, allow_utterance_fallback: bool = False) -> dict[str, Any]:
    from attenuate.data import load_manifest_records

    required = ("train_csv", "val_csv", "quick_val_csv", "test_csv")
    missing = [key for key in required if not paths.get(key)]
    if missing:
        raise ValueError(f"Campaign config missing required paths: {missing}")

    missing_files = [paths[key] for key in required if not Path(paths[key]).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing prepared manifests: {missing_files}")

    train_records = load_manifest_records(paths["train_csv"])
    val_records = load_manifest_records(paths["val_csv"])
    quick_records = load_manifest_records(paths["quick_val_csv"])
    test_records = load_manifest_records(paths["test_csv"])

    train_speakers = sorted({row.speaker_id for row in train_records})
    val_speakers = sorted({row.speaker_id for row in val_records})
    if not set(train_speakers).isdisjoint(val_speakers) and not allow_utterance_fallback:
        raise ValueError("train/val split is not speaker-disjoint")

    quick_utterances = {row.utterance_id for row in quick_records}
    val_utterances = {row.utterance_id for row in val_records}
    if not quick_utterances.issubset(val_utterances):
        raise ValueError("quick_val manifest must be a subset of val manifest")

    return {
        "counts": {
            "train": len(train_records),
            "val": len(val_records),
            "val_quick": len(quick_records),
            "test": len(test_records),
        },
        "speaker_disjoint": bool(set(train_speakers).isdisjoint(val_speakers)),
        "allow_utterance_fallback": bool(allow_utterance_fallback),
        "train_speakers": train_speakers,
        "val_speakers": val_speakers,
    }


def _ensure_dataset(paths: dict[str, str], dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    manifest_root = Path(paths["train_csv"]).parent
    raw_root = Path(paths.get("raw_root", "dataset/voicebank-demand/raw"))
    summary_path = manifest_root / "dataset_summary.json"
    if all(Path(paths[key]).exists() for key in ("train_csv", "val_csv", "quick_val_csv", "test_csv")):
        validation = _validate_prepared_manifests(paths, allow_utterance_fallback=bool(dataset_cfg.get("allow_utterance_fallback", False)))
        return {"prepared": False, "summary_path": summary_path.as_posix() if summary_path.exists() else None, "validation": validation}

    if not raw_root.exists():
        if not bool(dataset_cfg.get("auto_download", True)):
            raise FileNotFoundError(
                f"Prepared manifests are missing and raw dataset root does not exist: {raw_root}. "
                "Place VoiceBank-DEMAND raw audio there or pre-generate manifests."
            )
        download_summary = ensure_voicebank_raw_dataset(raw_root)
    else:
        download_summary = None

    manifest_root.mkdir(parents=True, exist_ok=True)
    summary = prepare_voicebank_dataset(
        source_root=raw_root,
        out_root=manifest_root,
        sample_rate=int(dataset_cfg.get("sample_rate", 16000)),
        val_speaker_fraction=float(dataset_cfg.get("val_speaker_fraction", 0.1)),
        val_quick_count=int(dataset_cfg.get("val_quick_count", 96)),
        seed=int(dataset_cfg.get("seed", 1337)),
        manifest_only=bool(dataset_cfg.get("manifest_only", False)),
        overwrite=bool(dataset_cfg.get("overwrite", False)),
        allow_utterance_fallback=bool(dataset_cfg.get("allow_utterance_fallback", False)),
    )
    validation = _validate_prepared_manifests(paths, allow_utterance_fallback=bool(dataset_cfg.get("allow_utterance_fallback", False)))
    return {
        "prepared": True,
        "summary_path": (manifest_root / "dataset_summary.json").as_posix(),
        "summary": summary,
        "download_summary": download_summary,
        "validation": validation,
    }


def _refinement_peer_map(best_base_runs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    if "erb_mrstft" in best_base_runs:
        mapping["erb_mrstft_aug"] = best_base_runs["erb_mrstft"]
    if "erb_mrstft_complex" in best_base_runs:
        mapping["erb_mrstft_full"] = best_base_runs["erb_mrstft_complex"]
    return mapping


def _analyze_refinement(
    *,
    run: dict[str, Any],
    protocol_reference: dict[str, Any],
    peer_reference: dict[str, Any] | None,
    gpu_memory_gb: float,
    total_epochs: int,
) -> dict[str, Any]:
    name = run["name"]
    cfg = run["config"]
    metrics = run["full_val"]
    diagnostics = run["summary"].get("diagnostics", {})
    signals = diagnostics.get("signals", {})
    objective_cfg = cfg.get("objective", {})
    augment_cfg = cfg.get("augment", {})
    issues: list[str] = []
    overrides: dict[str, Any] = {}
    rule = "none"
    rationale = "control_or_no_actionable_issue"

    if name in {"repo_baseline", "protocol_longctx"}:
        issues.append("control_experiment")
    else:
        if bool(signals.get("non_finite_detected")):
            issues.append("non_finite_outputs")
        if float(signals.get("selection_oscillation", 0.0)) > 0.03:
            issues.append("validation_instability")
        if bool(signals.get("clip_detected")):
            issues.append("clipping")
        if bool(signals.get("over_attenuation_suspected")):
            issues.append("over_attenuation_signal")
        if float(metrics["STOI"]) < float(protocol_reference["STOI"]) - 0.01 and float(metrics["SI_SDR"]) > float(protocol_reference["SI_SDR"]):
            issues.append("stoi_regression_vs_protocol")
        if augment_cfg.get("enabled") and peer_reference is not None and float(metrics["PESQ"]) < float(peer_reference["full_val"]["PESQ"]) - 0.01:
            issues.append("augmentation_penalty")
        if gpu_memory_gb >= 11.0 and int(cfg.get("segment_len", 64000)) < 96000 and int(run["summary"].get("best_epoch", 0)) < max(8, total_epochs):
            issues.append("context_headroom")

        if "validation_instability" in issues or "non_finite_outputs" in issues:
            rule = "stability_lr"
            overrides = {"lr": 5e-4, "warmup_ratio": 0.08}
            rationale = "Oscillating validation or non-finite behavior favors a calmer optimizer schedule."
        elif "stoi_regression_vs_protocol" in issues or "over_attenuation_signal" in issues:
            rule = "protect_intelligibility"
            overrides = {
                "objective": {
                    "mrstft_weight": 0.2,
                    "complex_weight": min(0.05, float(objective_cfg.get("complex_weight", 0.0))),
                    "high_snr_weight": 0.03,
                }
            }
            if float(objective_cfg.get("complex_weight", 0.0)) == 0.0:
                overrides["objective"]["complex_weight"] = 0.05
            rationale = "STOI lag while SI-SDR improves suggests over-suppression; reduce spectral aggressiveness."
        elif "augmentation_penalty" in issues:
            rule = "reduce_augment"
            overrides = {
                "augment": {
                    "enabled": True,
                    "probability": 0.5,
                    "max_ops": 1,
                    "clip_threshold_range": [0.5, 0.95],
                    "reverb_ir_ms_range": [15.0, 50.0],
                }
            }
            rationale = "Augmentation harms PESQ relative to the non-augment sibling; reduce severity and overlap."
        elif "context_headroom" in issues:
            rule = "longer_context"
            overrides = {"segment_len": 96000, "batch_size": 2}
            rationale = "GPU memory headroom exists and the run plateaued early; longer context is worth testing."

    return {
        "experiment": name,
        "issues": issues,
        "rule": rule,
        "overrides": overrides,
        "rationale": rationale,
        "diagnostics_verdict": diagnostics.get("verdict", "control"),
        "no_refinement_justified": rule == "none",
    }


def _run_experiment_stage(
    *,
    name: str,
    stage: str,
    base_cfg: dict[str, Any],
    paths: dict[str, str],
    output_root: Path,
    device: str,
    sample_rate: int,
) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg.update(
        {
            "train_csv": paths["train_csv"],
            "val_csv": paths["val_csv"],
            "quick_val_csv": paths["quick_val_csv"],
            "device": device,
            "run_dir": (output_root / stage / name).as_posix(),
            "checkpoint_out": (output_root / stage / name / "best.pt").as_posix(),
        }
    )
    summary, status = _run_or_resume_training(cfg)
    full_val = _evaluate_checkpoint(
        summary["best_checkpoint"],
        paths["val_csv"],
        device,
        sample_rate=sample_rate,
        frontend_cfg=cfg.get("frontend"),
        model_cfg=cfg.get("model"),
    )["aggregate"]
    return {"name": name, "stage": stage, "config": cfg, "summary": summary, "full_val": full_val, "status": status}


def _best_of_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    best = runs[0]
    for candidate in runs[1:]:
        if compare_metric_dicts(candidate["full_val"], best["full_val"]):
            best = candidate
    return best


def _export_model(script: str, checkpoint: str, out_path: Path, extra_args: list[str] | None = None) -> dict[str, Any]:
    command = [sys.executable, script, "--checkpoint", checkpoint, "--out", out_path.as_posix(), "--seq-len", "16000"]
    if extra_args:
        command.extend(extra_args)
    proc = subprocess.run(command, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "artifact": out_path.as_posix(),
        "success": proc.returncode == 0 and out_path.exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PESQ-first VoiceBank campaign.")
    parser.add_argument("--config", required=True, help="Campaign TOML/JSON config.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-experiments", type=int, default=None)
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    paths = dict(cfg.get("paths", {}))
    paths.setdefault("raw_root", "dataset/voicebank-demand/raw")
    defaults = dict(cfg.get("training_defaults", {}))
    promotion_cfg = dict(cfg.get("promotion", {}))
    surrogate_cfg = dict(cfg.get("surrogate", {}))
    dataset_cfg = dict(cfg.get("dataset", {}))
    evaluation_cfg = dict(cfg.get("evaluation", {}))
    experiments = list(cfg.get("experiments", []))
    if args.limit_experiments is not None:
        experiments = experiments[: args.limit_experiments]
    if not experiments:
        raise ValueError("No experiments defined in campaign config")

    sample_rate = int(defaults.get("sample_rate", 16000))
    output_root = Path(paths.get("output_root", "runs/pesq_campaign"))
    output_root.mkdir(parents=True, exist_ok=True)
    test_max_files = evaluation_cfg.get("test_max_files")
    lock = ProcessLock(output_root / "run_campaign.lock")
    if not lock.acquire():
        print(f"Another run_campaign instance already owns {output_root / 'run_campaign.lock'}; exiting.")
        return

    try:
        dataset_info = _ensure_dataset(paths, _deep_merge({"sample_rate": sample_rate}, dataset_cfg))
        _write_json(output_root / "dataset_validation.json", dataset_info)

        leaderboard_rows: list[dict[str, Any]] = []
        base_runs: list[dict[str, Any]] = []
        best_base_by_name: dict[str, dict[str, Any]] = {}

        for exp in experiments:
            exp_cfg = _deep_merge(defaults, exp)
            base_run = _run_experiment_stage(
                name=exp_cfg["name"],
                stage="base",
                base_cfg=exp_cfg,
                paths=paths,
                output_root=output_root,
                device=args.device,
                sample_rate=sample_rate,
            )
            base_runs.append(base_run)
            best_base_by_name[exp_cfg["name"]] = base_run
            leaderboard_rows.append(_summary_row(exp_cfg["name"], "base", base_run["full_val"], base_run["summary"]["best_checkpoint"]))

        if "protocol_longctx" not in best_base_by_name:
            raise ValueError("Campaign must define the protocol_longctx experiment")
        protocol_reference = best_base_by_name["protocol_longctx"]["full_val"]
        peer_map = _refinement_peer_map(best_base_by_name)
        available_vram = _gpu_memory_gb(args.device)
        total_epochs = int(defaults.get("epochs", 80))

        direction_summaries: list[dict[str, Any]] = []
        analysis_rows: list[dict[str, Any]] = []
        for base_run in base_runs:
            analysis = _analyze_refinement(
                run=base_run,
                protocol_reference=protocol_reference,
                peer_reference=peer_map.get(base_run["name"]),
                gpu_memory_gb=available_vram,
                total_epochs=total_epochs,
            )
            _write_json(output_root / "analysis" / f"{base_run['name']}.json", analysis)
            analysis_rows.append(analysis)

            candidate_runs = [base_run]
            refined_run = None
            if not analysis["no_refinement_justified"]:
                refined_cfg = _deep_merge(base_run["config"], analysis["overrides"])
                refined_cfg.pop("resume", None)
                refined_cfg.pop("init_checkpoint", None)
                refined_run = _run_experiment_stage(
                    name=base_run["name"],
                    stage="refine",
                    base_cfg=refined_cfg,
                    paths=paths,
                    output_root=output_root,
                    device=args.device,
                    sample_rate=sample_rate,
                )
                candidate_runs.append(refined_run)
                leaderboard_rows.append(
                    _summary_row(
                        base_run["name"],
                        "refine",
                        refined_run["full_val"],
                        refined_run["summary"]["best_checkpoint"],
                        {"refinement_rule": analysis["rule"]},
                    )
                )

            selected = _best_of_runs(candidate_runs)
            selected_test = _evaluate_checkpoint(
                selected["summary"]["best_checkpoint"],
                paths["test_csv"],
                args.device,
                sample_rate=sample_rate,
                max_files=test_max_files,
                frontend_cfg=selected["config"].get("frontend"),
                model_cfg=selected["config"].get("model"),
            )["aggregate"]
            direction_summary = {
                "name": base_run["name"],
                "base": base_run,
                "analysis": analysis,
                "refined": refined_run,
                "selected": selected,
                "selected_test": selected_test,
            }
            direction_summaries.append(direction_summary)
            leaderboard_rows.append(
                _summary_row(
                    base_run["name"],
                    "direction_best_val",
                    selected["full_val"],
                    selected["summary"]["best_checkpoint"],
                    {"refinement_rule": analysis["rule"]},
                )
            )
            leaderboard_rows.append(
                _summary_row(
                    base_run["name"],
                    "direction_best_test",
                    selected_test,
                    selected["summary"]["best_checkpoint"],
                    {"refinement_rule": analysis["rule"]},
                )
            )
        eligible_directions = [
            item
            for item in direction_summaries
            if (
                float(item["selected"]["full_val"]["STOI"]) >= float(protocol_reference["STOI"]) - float(promotion_cfg.get("max_stoi_drop", 0.015))
                and float(item["selected"]["full_val"]["SI_SDR"]) >= float(protocol_reference["SI_SDR"]) - float(promotion_cfg.get("max_sisdr_drop", 0.5))
                and not bool(item["selected"]["summary"].get("diagnostics", {}).get("signals", {}).get("non_finite_detected", False))
                and not bool(item["selected"]["summary"].get("diagnostics", {}).get("signals", {}).get("clip_detected", False))
            )
        ]
        if not eligible_directions:
            raise RuntimeError("No direction survived guardrails")

        eligible_directions = sorted(eligible_directions, key=lambda item: _sort_key(item["selected"]["full_val"]), reverse=True)
        winner_direction = eligible_directions[0]
        final_model = winner_direction["selected"]
        final_notes = {"stage2_applied": False, "stable_direction": True}

        if not args.skip_stage2 and surrogate_cfg.get("enabled", True):
            surrogate_root = output_root / "surrogate"
            sample_manifests = [paths["val_csv"]]
            if surrogate_cfg.get("include_train_manifest", True):
                sample_manifests.insert(0, paths["train_csv"])
            checkpoint_pool = [direction["selected"]["summary"]["best_checkpoint"] for direction in direction_summaries]
            samples_summary = build_surrogate_samples(
                sample_manifests,
                checkpoint_pool,
                surrogate_root / "samples.pt",
                device=args.device,
                sample_rate=sample_rate,
                max_rows_per_manifest=int(surrogate_cfg.get("max_rows_per_manifest", 96)),
            )
            surrogate_summary = train_surrogate(
                surrogate_root / "samples.pt",
                surrogate_root / "model",
                device=args.device,
                seed=int(defaults.get("seed", 1337)),
                epochs=int(surrogate_cfg.get("epochs", 20)),
                batch_size=int(surrogate_cfg.get("batch_size", 8)),
                lr=float(surrogate_cfg.get("lr", 1e-3)),
            )
            _write_json(output_root / "surrogate" / "samples_summary.json", samples_summary)
            if surrogate_summary["best_corr"] >= float(surrogate_cfg.get("min_corr", 0.85)):
                stage2_cfg = _deep_merge(winner_direction["selected"]["config"], surrogate_cfg.get("finetune_overrides", {}))
                stage2_cfg["init_checkpoint"] = winner_direction["selected"]["summary"]["best_checkpoint"]
                stage2_cfg["resume"] = None
                stage2_cfg["epochs"] = int(surrogate_cfg.get("finetune_epochs", 10))
                stage2_cfg["run_dir"] = (output_root / "stage2" / winner_direction["name"]).as_posix()
                stage2_cfg["checkpoint_out"] = (Path(stage2_cfg["run_dir"]) / "best.pt").as_posix()
                stage2_cfg["stage2"] = {
                    "enabled": True,
                    "checkpoint": surrogate_summary["best_checkpoint"],
                    "weight": float(surrogate_cfg.get("finetune_weight", 0.05)),
                    "warmup_epochs": int(surrogate_cfg.get("finetune_warmup_epochs", 2)),
                }
                stage2_summary, _stage2_status = _run_or_resume_training(stage2_cfg)
                stage2_val = _evaluate_checkpoint(
                    stage2_summary["best_checkpoint"],
                    paths["val_csv"],
                    args.device,
                    sample_rate=sample_rate,
                    frontend_cfg=stage2_cfg.get("frontend"),
                    model_cfg=stage2_cfg.get("model"),
                )["aggregate"]
                leaderboard_rows.append(_summary_row(winner_direction["name"], "stage2_val", stage2_val, stage2_summary["best_checkpoint"]))
                if (
                    float(stage2_val["PESQ"]) >= float(winner_direction["selected"]["full_val"]["PESQ"]) + float(surrogate_cfg.get("min_pesq_gain", 0.02))
                    and float(stage2_val["STOI"]) >= float(winner_direction["selected"]["full_val"]["STOI"]) - float(surrogate_cfg.get("max_stoi_drop", 0.01))
                    and float(stage2_val["SI_SDR"]) >= float(winner_direction["selected"]["full_val"]["SI_SDR"]) - float(surrogate_cfg.get("max_sisdr_drop", 0.5))
                ):
                    final_model = {
                        "name": winner_direction["name"],
                        "stage": "stage2",
                        "config": stage2_cfg,
                        "summary": stage2_summary,
                        "full_val": stage2_val,
                    }
                    final_notes["stage2_applied"] = True
                    winner_direction["stage2"] = {
                        "config": stage2_cfg,
                        "summary": stage2_summary,
                        "full_val": stage2_val,
                    }

        confirmation = None
        if not args.skip_confirmation:
            confirmation_cfg = _deep_merge(winner_direction["selected"]["config"], promotion_cfg.get("confirmation_overrides", {}))
            confirmation_cfg["seed"] = int(promotion_cfg.get("confirmation_seed", 2337))
            confirmation_cfg["resume"] = None
            confirmation_cfg["init_checkpoint"] = None
            confirmation_cfg["run_dir"] = (output_root / "confirmation" / winner_direction["name"]).as_posix()
            confirmation_cfg["checkpoint_out"] = (Path(confirmation_cfg["run_dir"]) / "best.pt").as_posix()
            confirmation_summary, _confirmation_status = _run_or_resume_training(confirmation_cfg)
            confirmation_val = _evaluate_checkpoint(
                confirmation_summary["best_checkpoint"],
                paths["val_csv"],
                args.device,
                sample_rate=sample_rate,
                frontend_cfg=confirmation_cfg.get("frontend"),
                model_cfg=confirmation_cfg.get("model"),
            )["aggregate"]
            confirmation = {"summary": confirmation_summary, "full_val": confirmation_val}
            final_notes["stable_direction"] = abs(float(confirmation_val["PESQ"]) - float(winner_direction["selected"]["full_val"]["PESQ"])) <= 0.03
            leaderboard_rows.append(_summary_row(winner_direction["name"], "confirmation_val", confirmation_val, confirmation_summary["best_checkpoint"]))

        final_test = _evaluate_checkpoint(
            final_model["summary"]["best_checkpoint"],
            paths["test_csv"],
            args.device,
            sample_rate=sample_rate,
            max_files=test_max_files,
            frontend_cfg=final_model["config"].get("frontend"),
            model_cfg=final_model["config"].get("model"),
        )["aggregate"]
        leaderboard_rows.append(_summary_row(final_model["name"], "final_test", final_test, final_model["summary"]["best_checkpoint"]))

        export_root = output_root / "exports"
        export_root.mkdir(parents=True, exist_ok=True)
        torchscript_export = _export_model("export_ts.py", final_model["summary"]["best_checkpoint"], export_root / "final_model.ts")
        onnx_export = _export_model(
            "export_onnx.py",
            final_model["summary"]["best_checkpoint"],
            export_root / "final_model.onnx",
            extra_args=["--prefer-dynamo"],
        )

        leaderboard_rows = sorted(leaderboard_rows, key=lambda row: (row["stage"], -row["PESQ"]))
        _write_leaderboard(leaderboard_rows, output_root / "leaderboard.csv")

        campaign_summary = {
            "dataset": dataset_info,
            "config_path": Path(args.config).as_posix(),
            "device": args.device,
            "gpu_memory_gb": available_vram,
            "protocol_reference": protocol_reference,
            "directions": direction_summaries,
            "analysis": analysis_rows,
            "winner_direction": winner_direction["name"],
            "final_model": {
                "name": final_model["name"],
                "stage": final_model["stage"],
                "checkpoint": final_model["summary"]["best_checkpoint"],
                "model": final_model["config"].get("model", {"kind": "atennuate"}),
                "val": final_model["full_val"],
                "test": final_test,
            },
            "confirmation": confirmation,
            "exports": {
                "torchscript": torchscript_export,
                "onnx": onnx_export,
            },
            "notes": final_notes,
            "rows": leaderboard_rows,
        }
        summary_path = output_root / "campaign_summary.json"
        _write_json(summary_path, campaign_summary)
        _write_json(output_root / "leaderboard.json", campaign_summary)

        if not args.skip_report:
            from generate_campaign_report import generate_campaign_report

            generate_campaign_report(
                summary_path=summary_path,
                out_path=Path("reports") / "voicebank_pesq_campaign.docx",
            )
    finally:
        lock.release()


if __name__ == "__main__":
    main()
