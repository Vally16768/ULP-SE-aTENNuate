from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sebench import MODEL_FAMILIES, MODEL_VARIANTS
from sebench.audio import load_mono_audio, save_mono_audio
from sebench.checkpoints import load_model_from_checkpoint
from sebench.runtime import require_cuda_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise a wav file using a multi-family checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt produced by train.py.")
    parser.add_argument("--input", required=True, help="Input noisy wav.")
    parser.add_argument("--output", required=True, help="Output enhanced wav.")
    parser.add_argument("--model-family", default=None, choices=list(MODEL_FAMILIES), help="Optional override for legacy checkpoints.")
    parser.add_argument("--variant", default=None, choices=list(MODEL_VARIANTS), help="Optional override for legacy checkpoints.")
    parser.add_argument("--postfilter-mode", default=None, choices=["none", "sg_residual_soft", "sg_input_floor"])
    parser.add_argument("--postfilter-preset", default=None, choices=["light", "medium", "aggressive"])
    parser.add_argument("--train-postfilter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--spectral-native-gate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.device = require_cuda_device(args.device)

    ckpt_path = Path(args.checkpoint)
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input wav not found: {in_path}")

    model, package = load_model_from_checkpoint(
        ckpt_path,
        device=args.device,
        model_family=args.model_family,
        variant=args.variant,
        postfilter_mode=args.postfilter_mode,
        postfilter_preset=args.postfilter_preset,
        train_postfilter=args.train_postfilter,
        spectral_native_gate=args.spectral_native_gate,
    )
    noisy, sr = load_mono_audio(in_path)
    with torch.no_grad():
        enhanced = model.denoise_single(noisy.unsqueeze(0).to(args.device)).squeeze(0).cpu()
    save_mono_audio(out_path, enhanced, sr)
    family = package.get("model_family", args.model_family or "atennuate")
    variant = package.get("variant", args.variant or "base")
    print(f"model_family={family} variant={variant}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
