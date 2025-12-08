import argparse
import csv
from pathlib import Path
from typing import List, Dict

import torch
import torchaudio

from attenuate.model import aTENNuate
from metrics.oracle import main as oracle_main
from metrics.metrics_logger import setup_metrics_logger


def load_mono_16k(path: Path, target_sr: int = 16000):
    """
    Încarcă audio, convertește la mono și resample la 16 kHz.
    Returnează (waveform, sample_rate).
    """
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0), target_sr  # (T,), sr


def generate_enhanced(
    model_ckpt: Path,
    manifest: Path,
    enhanced_dir: Path,
    max_files: int = None,
    device: str = "cpu",
) -> Path:
    """
    Rulează modelul peste perechile noisy/clean din manifest (noisy,clean)
    și salvează enhanced wavs în enhanced_dir. Întoarce calea către
    manifestul extins (clean,noisy,enhanced) pentru oracle.
    """
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    # model
    model = aTENNuate()
    state = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # citim manifestul (noisy,clean)
    rows_in: List[Dict[str, str]] = []
    with manifest.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "noisy" not in reader.fieldnames or "clean" not in reader.fieldnames:
            raise ValueError("Manifest must contain columns: noisy, clean")
        for r in reader:
            rows_in.append(r)

    if max_files is not None:
        rows_in = rows_in[:max_files]

    rows_out: List[Dict[str, str]] = []

    for r in rows_in:
        noisy_path = Path(r["noisy"])
        clean_path = Path(r["clean"])
        if not noisy_path.exists():
            raise FileNotFoundError(noisy_path)
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)

        wav_noisy, sr = load_mono_16k(noisy_path)
        wav_noisy = wav_noisy.unsqueeze(0)  # (1, T)

        with torch.no_grad():
            wav_noisy = wav_noisy.to(device)
            enhanced = model.denoise_single(wav_noisy)  # (1, T)
            enhanced = enhanced.squeeze(0).cpu()

        enh_name = noisy_path.stem + "_enh.wav"
        enh_path = enhanced_dir / enh_name
        torchaudio.save(enh_path.as_posix(), enhanced.unsqueeze(0), sr)

        rows_out.append({
            "clean": clean_path.as_posix(),
            "noisy": noisy_path.as_posix(),
            "enhanced": enh_path.as_posix(),
        })

    # scriem manifest pentru oracle (clean,noisy,enhanced)
    oracle_manifest = enhanced_dir / "manifest_oracle.csv"
    with oracle_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clean", "noisy", "enhanced"])
        writer.writeheader()
        writer.writerows(rows_out)

    return oracle_manifest


def main():
    ap = argparse.ArgumentParser(
        description="Rulează metricile intrusive (oracle) pentru un model."
    )
    ap.add_argument(
        "--checkpoint", required=True, help="Checkpoint model (.pt)."
    )
    ap.add_argument(
        "--manifest", required=True,
        help="CSV cu coloane noisy,clean (VoiceBank-DEMAND test)."
    )
    ap.add_argument(
        "--enhanced-dir", required=True,
        help="Director unde se salvează enhanced wavs + manifest_oracle.csv."
    )
    ap.add_argument(
        "--oracle-json", required=True,
        help="Fișier JSON de ieșire pentru oracle."
    )
    ap.add_argument(
        "--max-files", type=int, default=None,
        help="Număr maxim de fișiere de evaluat (debug)."
    )
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = ap.parse_args()

    setup_metrics_logger()  # pentru log-urile metricilor

    ckpt_path = Path(args.checkpoint)
    manifest = Path(args.manifest)
    enhanced_dir = Path(args.enhanced_dir)
    oracle_json = Path(args.oracle_json)

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    oracle_manifest = generate_enhanced(
        ckpt_path,
        manifest,
        enhanced_dir,
        max_files=args.max_files,
        device=args.device,
    )

    # rulează oracle (PESQ, STOI, SI-SDR, Delta-SNR) și scrie JSON
    oracle_main(
        manifest_csv=oracle_manifest.as_posix(),
        out_json=oracle_json.as_posix(),
        thresholds=None,
    )


if __name__ == "__main__":
    main()
