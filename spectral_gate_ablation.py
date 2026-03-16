import argparse
import json
from pathlib import Path

import torch

from attenuate.checkpoints import load_model_config_file, load_state_dict_file
from attenuate.eval_runtime import (
    evaluate_frontend_on_manifest,
    evaluate_model_on_manifest,
    evaluate_noisy_baseline,
)
from attenuate.model import build_model


PRESETS = {
    "mild": {
        "kind": "spectral_gate",
        "noise_quantile": 0.10,
        "threshold_scale": 1.15,
        "mask_slope": 8.0,
        "mask_floor": 0.18,
    },
    "default": {
        "kind": "spectral_gate",
        "noise_quantile": 0.15,
        "threshold_scale": 1.25,
        "mask_slope": 10.0,
        "mask_floor": 0.10,
    },
    "strong": {
        "kind": "spectral_gate",
        "noise_quantile": 0.20,
        "threshold_scale": 1.40,
        "mask_slope": 12.0,
        "mask_floor": 0.05,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare spectral gating before aTENNuate.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["mild", "default", "strong"],
        choices=sorted(PRESETS.keys()),
        help="Spectral gate presets to compare.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest)
    out_path = Path(args.output_json)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    model_cfg = load_model_config_file(ckpt_path, fallback={"kind": "atennuate"})
    model = build_model(model_cfg)
    model.load_state_dict(load_state_dict_file(ckpt_path, map_location="cpu"))
    model.to(args.device)
    model.eval()

    results: dict[str, dict] = {}
    results["noisy_baseline"] = evaluate_noisy_baseline(
        manifest_path,
        sample_rate=args.sample_rate,
        max_files=args.max_files,
    )["aggregate"]
    results["model_only"] = evaluate_model_on_manifest(
        model,
        manifest_path,
        device=args.device,
        sample_rate=args.sample_rate,
        max_files=args.max_files,
        desc="model_only",
    )["aggregate"]

    for preset in args.presets:
        frontend_cfg = PRESETS[preset]
        results[f"spectral_gate_only::{preset}"] = evaluate_frontend_on_manifest(
            manifest_path,
            frontend_cfg=frontend_cfg,
            sample_rate=args.sample_rate,
            max_files=args.max_files,
            desc=f"spectral_gate_only::{preset}",
        )["aggregate"]
        results[f"spectral_gate_plus_model::{preset}"] = evaluate_model_on_manifest(
            model,
            manifest_path,
            device=args.device,
            sample_rate=args.sample_rate,
            max_files=args.max_files,
            desc=f"spectral_gate_plus_model::{preset}",
            frontend_cfg=frontend_cfg,
        )["aggregate"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
