from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attenuate.checkpoints import load_model_config_file, load_state_dict_file
from attenuate.eval_runtime import evaluate_model_on_manifest, evaluate_noisy_baseline
from attenuate.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a noisy/clean manifest and compute delta metrics vs noisy baseline.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-kind", default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    manifest = Path(args.manifest)
    out_json = Path(args.out_json)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    model_cfg = load_model_config_file(ckpt_path, fallback={"kind": args.model_kind or "atennuate"})
    if args.model_kind is not None:
        model_cfg["kind"] = args.model_kind
    sample_rate = int(args.sample_rate or model_cfg.get("sample_rate", 16000))

    model = build_model(model_cfg)
    model.load_state_dict(load_state_dict_file(ckpt_path, map_location="cpu"))
    model.to(args.device)
    model.eval()

    noisy_baseline = evaluate_noisy_baseline(
        manifest,
        sample_rate=sample_rate,
        max_files=args.max_files,
    )
    evaluated = evaluate_model_on_manifest(
        model,
        manifest,
        device=args.device,
        sample_rate=sample_rate,
        max_files=args.max_files,
        desc=manifest.stem,
        frontend_cfg=None,
    )

    aggregate = dict(evaluated["aggregate"])
    baseline = dict(noisy_baseline["aggregate"])
    delta = {
        "delta_PESQ": float(aggregate["PESQ"] - baseline["PESQ"]),
        "delta_STOI": float(aggregate["STOI"] - baseline["STOI"]),
        "delta_SI_SDR": float(aggregate["SI_SDR"] - baseline["SI_SDR"]),
        "delta_DELTA_SNR": float(aggregate["DELTA_SNR"] - baseline["DELTA_SNR"]),
    }
    payload = {
        "checkpoint": ckpt_path.as_posix(),
        "manifest": manifest.as_posix(),
        "sample_rate": sample_rate,
        "model_config": model_cfg,
        "aggregate": aggregate,
        "noisy_baseline": baseline,
        "delta": delta,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
