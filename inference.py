import argparse
from pathlib import Path

import torch
import torchaudio

from attenuate.checkpoints import load_model_config_file, load_state_dict_file
from attenuate.model import architecture_summary, build_model


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise a noisy wav file with aTENNuate.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt).")
    parser.add_argument("--input", required=True, help="Input noisy wav.")
    parser.add_argument("--output", required=True, help="Output enhanced wav.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-kind", default=None, help="Optional model override if checkpoint has no metadata.")
    parser.add_argument("--sample-rate", type=int, default=None, help="Optional resample rate. Defaults to checkpoint sample_rate.")
    parser.add_argument("--spectral-gate", action="store_true", help="Apply spectral gating before the model.")
    parser.add_argument("--gate-noise-quantile", type=float, default=0.15)
    parser.add_argument("--gate-threshold-scale", type=float, default=1.25)
    parser.add_argument("--gate-mask-slope", type=float, default=10.0)
    parser.add_argument("--gate-mask-floor", type=float, default=0.10)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input wav not found: {in_path}")

    device = args.device
    print(f"Using device: {device}")

    model_cfg = load_model_config_file(ckpt_path, fallback={"kind": args.model_kind or "atennuate"})
    if args.model_kind is not None:
        model_cfg["kind"] = args.model_kind
    summary = architecture_summary(model_cfg)
    model = build_model(model_cfg)
    model.load_state_dict(load_state_dict_file(ckpt_path, map_location="cpu"))
    model.to(device)
    model.eval()

    sample_rate = int(args.sample_rate or model_cfg.get("sample_rate", summary.get("sample_rate", 16000)))
    noisy, sr = load_mono_target(in_path, sample_rate)
    noisy = noisy.unsqueeze(0)
    frontend_cfg = frontend_from_args(args)

    with torch.no_grad():
        enhanced = model.denoise_single(
            noisy.to(device),
            frontend_cfg=frontend_cfg,
            sample_rate=sr,
        ).squeeze(0).cpu()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(out_path.as_posix(), enhanced.unsqueeze(0), sr)
    print(f"Saved denoised audio -> {out_path}")


if __name__ == "__main__":
    main()
