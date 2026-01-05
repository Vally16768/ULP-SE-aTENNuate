import argparse
from pathlib import Path

import torch
import torchaudio

from attenuate.model import aTENNuate


def load_mono_16k(path: Path, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0), target_sr  # (T,), sr


def main():
    ap = argparse.ArgumentParser(description="Denoise un fișier audio folosind aTENNuate.")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint .pt (state_dict) al modelului.")
    ap.add_argument("--input", required=True, help="Fișier wav zgomotos.")
    ap.add_argument("--output", required=True, help="Fișier wav de ieșire denoisat.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input wav not found: {in_path}")

    device = args.device
    print(f"Using device: {device}")

    # model
    model = aTENNuate()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # audio
    noisy, sr = load_mono_16k(in_path)
    noisy = noisy.unsqueeze(0)  # (1, T)

    with torch.no_grad():
        noisy = noisy.to(device)
        denoised = model.denoise_single(noisy)  # (1, T)
        denoised = denoised.squeeze(0).cpu()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(out_path.as_posix(), denoised.unsqueeze(0), sr)
    print(f"Saved denoised audio -> {out_path}")


if __name__ == "__main__":
    main()
