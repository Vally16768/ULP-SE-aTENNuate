import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio

from attenuate.checkpoints import load_model_config_file, load_state_dict_file
from attenuate.model import architecture_summary, build_model
from metrics.metrics_logger import setup_metrics_logger
from metrics.oracle import main as oracle_main


def load_mono_target(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr


def frontend_from_args(args: argparse.Namespace) -> dict | None:
    if not args.spectral_gate:
        return None
    return {
        "kind": "spectral_gate",
        "noise_quantile": args.gate_noise_quantile,
        "threshold_scale": args.gate_threshold_scale,
        "mask_slope": args.gate_mask_slope,
        "mask_floor": args.gate_mask_floor,
    }


def generate_enhanced(
    model_ckpt: Path,
    manifest: Path,
    enhanced_dir: Path,
    max_files: int | None = None,
    device: str = "cpu",
    frontend_cfg: dict | None = None,
    model_kind: str | None = None,
    sample_rate: int | None = None,
) -> Path:
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_model_config_file(model_ckpt, fallback={"kind": model_kind or "atennuate"})
    if model_kind is not None:
        model_cfg["kind"] = model_kind
    summary = architecture_summary(model_cfg)
    model = build_model(model_cfg)
    model.load_state_dict(load_state_dict_file(model_ckpt, map_location="cpu"))
    model.to(device)
    model.eval()
    target_sr = int(sample_rate or model_cfg.get("sample_rate", summary.get("sample_rate", 16000)))

    rows_in: List[Dict[str, str]] = []
    with manifest.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "noisy" not in reader.fieldnames or "clean" not in reader.fieldnames:
            raise ValueError("Manifest must contain columns: noisy, clean")
        rows_in.extend(reader)

    if max_files is not None:
        rows_in = rows_in[:max_files]

    rows_out: List[Dict[str, str]] = []
    for row in rows_in:
        noisy_path = Path(row["noisy"])
        clean_path = Path(row["clean"])
        if not noisy_path.exists():
            raise FileNotFoundError(noisy_path)
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)

        wav_noisy, sr = load_mono_target(noisy_path, target_sr)
        wav_noisy = wav_noisy.unsqueeze(0)

        with torch.no_grad():
            enhanced = model.denoise_single(
                wav_noisy.to(device),
                frontend_cfg=frontend_cfg,
                sample_rate=sr,
            ).squeeze(0).cpu()

        enh_name = noisy_path.stem + "_enh.wav"
        enh_path = enhanced_dir / enh_name
        torchaudio.save(enh_path.as_posix(), enhanced.unsqueeze(0), sr)
        rows_out.append(
            {
                "clean": clean_path.as_posix(),
                "noisy": noisy_path.as_posix(),
                "enhanced": enh_path.as_posix(),
            }
        )

    oracle_manifest = enhanced_dir / "manifest_oracle.csv"
    with oracle_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clean", "noisy", "enhanced"])
        writer.writeheader()
        writer.writerows(rows_out)
    return oracle_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run intrusive oracle metrics for a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt).")
    parser.add_argument("--manifest", required=True, help="CSV with noisy,clean columns.")
    parser.add_argument("--enhanced-dir", required=True, help="Output directory for enhanced wavs.")
    parser.add_argument("--oracle-json", required=True, help="Output JSON for oracle metrics.")
    parser.add_argument("--max-files", type=int, default=None, help="Evaluate only the first N files.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-kind", default=None, help="Optional model override if checkpoint has no metadata.")
    parser.add_argument("--sample-rate", type=int, default=None, help="Optional resample rate. Defaults to checkpoint sample_rate.")
    parser.add_argument("--spectral-gate", action="store_true", help="Apply spectral gating before the model.")
    parser.add_argument("--gate-noise-quantile", type=float, default=0.15)
    parser.add_argument("--gate-threshold-scale", type=float, default=1.25)
    parser.add_argument("--gate-mask-slope", type=float, default=10.0)
    parser.add_argument("--gate-mask-floor", type=float, default=0.10)
    args = parser.parse_args()

    setup_metrics_logger()

    ckpt_path = Path(args.checkpoint)
    manifest = Path(args.manifest)
    enhanced_dir = Path(args.enhanced_dir)
    oracle_json = Path(args.oracle_json)

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    oracle_manifest = generate_enhanced(
        model_ckpt=ckpt_path,
        manifest=manifest,
        enhanced_dir=enhanced_dir,
        max_files=args.max_files,
        device=args.device,
        frontend_cfg=frontend_from_args(args),
        model_kind=args.model_kind,
        sample_rate=args.sample_rate,
    )

    oracle_main(
        manifest_csv=oracle_manifest.as_posix(),
        out_json=oracle_json.as_posix(),
        thresholds=None,
    )


if __name__ == "__main__":
    main()
