from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mono_audio(path: Path | str, sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(Path(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate
    return wav.squeeze(0).float(), sr


def save_mono_audio(path: Path | str, waveform: torch.Tensor, sample_rate: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = waveform.detach().cpu().float().unsqueeze(0)
    torchaudio.save(path.as_posix(), wav, sample_rate)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    padding = (multiple - x.shape[-1] % multiple) % multiple
    if padding == 0:
        return x, 0
    return F.pad(x, (0, padding)), padding


def clamp_audio(x: torch.Tensor, limit: float = 0.999) -> torch.Tensor:
    return torch.clamp(x, -limit, limit)


def speaker_id_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]


def utterance_id_from_path(path: Path | str) -> str:
    return Path(path).stem


def stack_with_lengths(signals: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    signals = [sig.float() for sig in signals]
    if not signals:
        raise ValueError("signals must not be empty")
    lengths = torch.tensor([sig.shape[-1] for sig in signals], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch = torch.stack([F.pad(sig, (0, max_len - sig.shape[-1])) for sig in signals], dim=0)
    return batch, lengths


def crop_or_pad_pair(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    segment_len: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noisy.shape != clean.shape:
        n = min(noisy.shape[-1], clean.shape[-1])
        noisy = noisy[:n]
        clean = clean[:n]

    if noisy.shape[-1] >= segment_len:
        start = rng.randrange(0, noisy.shape[-1] - segment_len + 1)
        return noisy[start : start + segment_len], clean[start : start + segment_len]

    pad = segment_len - noisy.shape[-1]
    return F.pad(noisy, (0, pad)), F.pad(clean, (0, pad))


def snr_db(clean: torch.Tensor, noisy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if clean.shape != noisy.shape:
        n = min(clean.shape[-1], noisy.shape[-1])
        clean = clean[..., :n]
        noisy = noisy[..., :n]
    signal = torch.sum(clean * clean, dim=-1)
    noise = torch.sum((clean - noisy) ** 2, dim=-1)
    return 10.0 * torch.log10((signal + eps) / (noise + eps))


def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)


def normalize_peak(x: torch.Tensor, peak: float = 0.98) -> torch.Tensor:
    peak_now = x.abs().amax().clamp_min(1e-8)
    return x * (peak / peak_now)


def db_to_amplitude(db: float) -> float:
    return math.pow(10.0, db / 20.0)


def spectral_gate_waveform(
    waveform: torch.Tensor,
    *,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    noise_quantile: float = 0.15,
    threshold_scale: float = 1.25,
    mask_slope: float = 10.0,
    mask_floor: float = 0.10,
    smooth_freq_bins: int = 5,
    smooth_time_frames: int = 9,
) -> torch.Tensor:
    """
    Simple stationary-noise spectral gate estimated directly from the noisy signal.

    waveform: (T,) or (B, T)
    returns: same shape as input
    """
    if waveform.ndim == 1:
        batch = waveform.unsqueeze(0)
        squeeze = True
    elif waveform.ndim == 2:
        batch = waveform
        squeeze = False
    else:
        raise ValueError(f"Expected waveform with shape (T,) or (B, T), got {tuple(waveform.shape)}")

    if not 0.0 < noise_quantile < 1.0:
        raise ValueError("noise_quantile must be in (0, 1)")
    if n_fft <= 0 or hop_length <= 0 or win_length <= 0:
        raise ValueError("n_fft, hop_length and win_length must be positive")

    work = batch.float()
    original_len = work.shape[-1]
    window = torch.hann_window(win_length, device=work.device, dtype=work.dtype)
    spec = torch.stft(
        work,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    mag = spec.abs()
    noise_mag = torch.quantile(mag, noise_quantile, dim=-1, keepdim=True)
    threshold = noise_mag * float(threshold_scale)
    ratio = (mag - threshold) / threshold.clamp_min(1e-6)
    mask = torch.sigmoid(float(mask_slope) * ratio)
    mask = mask * (1.0 - float(mask_floor)) + float(mask_floor)

    if smooth_freq_bins > 1 or smooth_time_frames > 1:
        freq_kernel = max(1, int(smooth_freq_bins))
        time_kernel = max(1, int(smooth_time_frames))
        if freq_kernel % 2 == 0:
            freq_kernel += 1
        if time_kernel % 2 == 0:
            time_kernel += 1
        mask = F.avg_pool2d(
            mask.unsqueeze(1),
            kernel_size=(freq_kernel, time_kernel),
            stride=1,
            padding=(freq_kernel // 2, time_kernel // 2),
        ).squeeze(1)

    enhanced = torch.istft(
        spec * mask,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=original_len,
    )
    return enhanced.squeeze(0) if squeeze else enhanced


def apply_input_frontend(
    waveform: torch.Tensor,
    frontend_cfg: dict | None = None,
    *,
    sample_rate: int = 16000,
) -> torch.Tensor:
    if not frontend_cfg:
        return waveform
    kind = str(frontend_cfg.get("kind", "")).strip().lower()
    if not kind or kind == "none":
        return waveform
    if kind == "spectral_gate":
        cfg = dict(frontend_cfg)
        cfg.pop("kind", None)
        return spectral_gate_waveform(waveform, sample_rate=sample_rate, **cfg)
    raise ValueError(f"Unsupported frontend kind: {kind}")
