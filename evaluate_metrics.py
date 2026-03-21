from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mlflow
import torch

from sebench import MODEL_FAMILIES, MODEL_VARIANTS
from sebench.audio import load_mono_audio, save_mono_audio
from sebench.checkpoints import load_model_from_checkpoint
from sebench.data import read_pair_manifest
from sebench.models import dynamic_quantize_metricgan
from sebench.runtime import require_cuda_device
from sebench.training import evaluate_manifest


def export_enhanced_wavs(
    checkpoint: Path,
    manifest: Path,
    enhanced_dir: Path,
    device: str,
    model_family: str | None = None,
    variant: str | None = None,
    postfilter_mode: str | None = None,
    postfilter_preset: str | None = None,
    train_postfilter: bool | None = None,
    spectral_native_gate: bool | None = None,
    max_files: int | None = None,
) -> Path:
    model, _ = load_model_from_checkpoint(
        checkpoint,
        device=device,
        model_family=model_family,
        variant=variant,
        postfilter_mode=postfilter_mode,
        postfilter_preset=postfilter_preset,
        train_postfilter=train_postfilter,
        spectral_native_gate=spectral_native_gate,
    )
    rows = read_pair_manifest(manifest)
    if max_files is not None:
        rows = rows[:max_files]
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, str]] = []

    for row in rows:
        noisy, sr = load_mono_audio(row.noisy)
        with torch.no_grad():
            enhanced = model.denoise_single(noisy.unsqueeze(0).to(device)).squeeze(0).cpu()
        enh_path = enhanced_dir / f"{row.noisy.stem}_enh.wav"
        save_mono_audio(enh_path, enhanced, sr)
        out_rows.append(
            {
                "clean": row.clean.as_posix(),
                "noisy": row.noisy.as_posix(),
                "enhanced": enh_path.as_posix(),
            }
        )

    oracle_manifest = enhanced_dir / "manifest_oracle.csv"
    with oracle_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clean", "noisy", "enhanced"])
        writer.writeheader()
        writer.writerows(out_rows)
    return oracle_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate intrusive metrics for a checkpoint on a noisy/clean manifest.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint model (.pt).")
    parser.add_argument("--manifest", required=True, help="CSV with columns noisy,clean.")
    parser.add_argument("--oracle-json", required=True, help="Output JSON file with aggregate metrics.")
    parser.add_argument("--enhanced-dir", default=None, help="Optional directory to export enhanced wavs.")
    parser.add_argument("--model-family", default=None, choices=list(MODEL_FAMILIES))
    parser.add_argument("--variant", default=None, choices=list(MODEL_VARIANTS))
    parser.add_argument("--postfilter-mode", default=None, choices=["none", "sg_residual_soft", "sg_input_floor"])
    parser.add_argument("--postfilter-preset", default=None, choices=["light", "medium", "aggressive"])
    parser.add_argument("--train-postfilter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--spectral-native-gate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--no-dnsmos", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--quantize-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mlflow-run-id", default=None)
    parser.add_argument("--metric-prefix", default="eval")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.device = require_cuda_device(args.device)

    ckpt_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest)
    oracle_json = Path(args.oracle_json)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    model, _ = load_model_from_checkpoint(
        ckpt_path,
        device=args.device,
        model_family=args.model_family,
        variant=args.variant,
        postfilter_mode=args.postfilter_mode,
        postfilter_preset=args.postfilter_preset,
        train_postfilter=args.train_postfilter,
        spectral_native_gate=args.spectral_native_gate,
    )
    if args.quantize_dynamic:
        if args.model_family == "metricgan_plus":
            raise ValueError(
                "--quantize-dynamic is not supported with metricgan_plus adapter checkpoints here. "
                "Use campaign.py teacher audit phases for the 16 kHz teacher benchmark."
            )
        model = dynamic_quantize_metricgan(model)
    metrics = evaluate_manifest(
        model,
        manifest_path.as_posix(),
        args.device,
        sample_rate=args.sample_rate,
        compute_dnsmos=not args.no_dnsmos,
        max_files=args.max_files,
        sample_dir=args.enhanced_dir,
        sample_count=3 if args.enhanced_dir else 0,
    )
    oracle_json.parent.mkdir(parents=True, exist_ok=True)
    with oracle_json.open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    if args.enhanced_dir:
        export_enhanced_wavs(
            ckpt_path,
            manifest_path,
            Path(args.enhanced_dir),
            args.device,
            model_family=args.model_family,
            variant=args.variant,
            postfilter_mode=args.postfilter_mode,
            postfilter_preset=args.postfilter_preset,
            train_postfilter=args.train_postfilter,
            spectral_native_gate=args.spectral_native_gate,
            max_files=args.max_files,
        )

    if args.mlflow_run_id:
        with mlflow.start_run(run_id=args.mlflow_run_id):
            mlflow.log_metrics(
                {f"{args.metric_prefix}/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))}
            )
            mlflow.log_artifact(oracle_json.as_posix(), artifact_path="reports")

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
