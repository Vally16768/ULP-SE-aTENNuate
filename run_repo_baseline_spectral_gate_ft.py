import argparse
import json
import tomllib
from pathlib import Path

import torch

from attenuate.checkpoints import load_model_config_file, load_state_dict_file
from attenuate.engine import run_training
from attenuate.eval_runtime import evaluate_model_on_manifest, write_rows_csv
from attenuate.model import build_model


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_model(checkpoint: Path, device: str) -> torch.nn.Module:
    model_cfg = load_model_config_file(checkpoint, fallback={"kind": "atennuate"})
    model = build_model(model_cfg)
    model.load_state_dict(load_state_dict_file(checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()
    return model


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    manifest_csv: Path,
    device: str,
    sample_rate: int,
    frontend_cfg: dict | None,
    desc: str,
    out_dir: Path,
) -> dict:
    model = load_model(checkpoint, device=device)
    result = evaluate_model_on_manifest(
        model,
        manifest_csv=manifest_csv,
        device=device,
        sample_rate=sample_rate,
        frontend_cfg=frontend_cfg,
        desc=desc,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{desc}_summary.json").write_text(
        json.dumps(result["aggregate"], indent=2),
        encoding="utf-8",
    )
    write_rows_csv(result["rows"], out_dir / f"{desc}_rows.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate repo_baseline + spectral gate fine-tune.")
    parser.add_argument("--config", default="experiments/repo_baseline_spectral_gate_ft.toml")
    parser.add_argument("--test-manifest", default="dataset/voicebank-demand/16k/test.csv")
    parser.add_argument("--threshold-pesq", type=float, default=2.4100624991273416)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None, help="Optional last_train_state.pt to resume training.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    config["device"] = args.device
    if args.resume:
        config["resume"] = args.resume

    summary = run_training(config)
    run_dir = Path(summary["config"]["run_dir"])
    eval_dir = run_dir / "test_eval"
    manifest_csv = Path(args.test_manifest)
    if not manifest_csv.exists():
        raise FileNotFoundError(manifest_csv)

    frontend_cfg = dict(summary["config"].get("frontend", {}))
    sample_rate = int(summary["config"].get("sample_rate", 16000))
    baseline_ckpt = Path(summary["config"]["init_checkpoint"])
    finetuned_ckpt = Path(summary["best_checkpoint"])

    baseline_result = evaluate_checkpoint(
        checkpoint=baseline_ckpt,
        manifest_csv=manifest_csv,
        device=args.device,
        sample_rate=sample_rate,
        frontend_cfg=frontend_cfg,
        desc="baseline_plus_gate",
        out_dir=eval_dir,
    )
    finetuned_result = evaluate_checkpoint(
        checkpoint=finetuned_ckpt,
        manifest_csv=manifest_csv,
        device=args.device,
        sample_rate=sample_rate,
        frontend_cfg=frontend_cfg,
        desc="finetuned_plus_gate",
        out_dir=eval_dir,
    )

    baseline_pesq = float(baseline_result["aggregate"]["PESQ"])
    finetuned_pesq = float(finetuned_result["aggregate"]["PESQ"])
    comparison = {
        "config_path": config_path.as_posix(),
        "training_summary": summary,
        "baseline_plus_gate": baseline_result["aggregate"],
        "finetuned_plus_gate": finetuned_result["aggregate"],
        "reference_threshold_pesq": float(args.threshold_pesq),
        "delta_vs_baseline_plus_gate": finetuned_pesq - baseline_pesq,
        "delta_vs_reference_threshold": finetuned_pesq - float(args.threshold_pesq),
        "beats_reference_threshold": finetuned_pesq > float(args.threshold_pesq),
    }
    (eval_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
