from __future__ import annotations

from pathlib import Path

import librosa
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from attenuate.audio import apply_input_frontend
from attenuate.losses import build_erb_filterbank


class PercepNetClass(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        sample_rate: int = 16000,
        n_fft: int | None = None,
        hop_length: int | None = None,
        win_length: int | None = None,
        n_bands: int = 32,
        hidden_size: int = 96,
        num_layers: int = 2,
        mask_gain: float = 1.2,
        residual_gain: float = 0.10,
        stft_center: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("PercepNetClass currently supports mono input only")

        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length or max(80, self.sample_rate // 100))
        self.win_length = int(win_length or max(160, self.sample_rate // 50))
        self.n_fft = int(n_fft or 1 << (self.win_length - 1).bit_length())
        self.n_bands = int(n_bands)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.mask_gain = float(mask_gain)
        self.residual_gain = float(residual_gain)
        self.stft_center = bool(stft_center)
        self.frame_len = self.win_length
        self.hop_len = self.hop_length
        self.analysis_padding = 0 if self.stft_center else max(self.win_length - self.hop_length, 0)

        window = torch.hamming_window(self.win_length)
        filterbank, _ = build_erb_filterbank(
            n_fft=self.n_fft,
            sample_rate=self.sample_rate,
            n_bands=self.n_bands,
        )
        band_to_freq = filterbank / filterbank.sum(dim=1, keepdim=True).clamp_min(1e-6)

        self.register_buffer("window", window, persistent=False)
        self.register_buffer("filterbank", filterbank, persistent=False)
        self.register_buffer("band_to_freq", band_to_freq, persistent=False)

        self.feature_proj = nn.Sequential(
            nn.Linear(self.n_bands * 2, self.hidden_size),
            nn.SiLU(),
        )
        self.pre_gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.main_gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.mask_head = nn.Linear(self.hidden_size, self.n_bands)
        self.residual_head = nn.Linear(self.hidden_size, self.n_bands)

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=self.stft_center,
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(spec.device),
            center=self.stft_center,
            length=length,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError("Expected input shaped (batch, 1, time)")
        waveform = input[:, 0, :]
        length = waveform.shape[-1]
        padded_waveform = waveform
        if self.analysis_padding > 0:
            padded_waveform = F.pad(padded_waveform, (self.analysis_padding, self.analysis_padding))
        if padded_waveform.shape[-1] < self.win_length:
            padded_waveform = F.pad(padded_waveform, (0, self.win_length - padded_waveform.shape[-1]))
        noisy_spec = self._stft(padded_waveform)
        noisy_mag = noisy_spec.abs().clamp_min(1e-7)
        noisy_phase = torch.angle(noisy_spec)

        erb = torch.einsum("bft,fm->btm", torch.log1p(noisy_mag), self.filterbank.to(noisy_mag.device))
        delta = torch.diff(erb, dim=1, prepend=erb[:, :1, :])
        feats = torch.cat([erb, delta], dim=-1)
        feats = self.feature_proj(feats)
        feats, _ = self.pre_gru(feats)
        feats, _ = self.main_gru(feats)

        band_mask = torch.sigmoid(self.mask_head(feats)) * self.mask_gain
        band_residual = torch.tanh(self.residual_head(feats)) * self.residual_gain
        band_mask = (band_mask + band_residual).clamp(0.0, 1.5)

        freq_mask = torch.einsum("btm,fm->btf", band_mask, self.band_to_freq.to(band_mask.device)).transpose(1, 2)
        enhanced_mag = noisy_mag * freq_mask.clamp_min(0.0)
        enhanced_spec = torch.polar(enhanced_mag, noisy_phase)
        enhanced = self._istft(enhanced_spec, length=padded_waveform.shape[-1])
        if self.analysis_padding > 0:
            enhanced = enhanced[..., self.analysis_padding : self.analysis_padding + length]
        else:
            enhanced = enhanced[..., :length]
        return enhanced.unsqueeze(1)

    def denoise_single(
        self,
        noisy: torch.Tensor,
        frontend_cfg: dict | None = None,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        if noisy.ndim != 2:
            raise ValueError("noisy input should be shaped (batch, length)")
        noisy = apply_input_frontend(noisy, frontend_cfg, sample_rate=sample_rate)
        enhanced = self.forward(noisy.unsqueeze(1))
        return enhanced.squeeze(1)

    def denoise_multiple(
        self,
        noisy_samples: list[torch.Tensor],
        frontend_cfg: dict | None = None,
        sample_rate: int = 16000,
    ) -> list[torch.Tensor]:
        audio_lens = [noisy.shape[-1] for noisy in noisy_samples]
        max_len = max(audio_lens)
        target_len = max(max_len, self.win_length)
        batched = torch.stack([F.pad(noisy, (0, target_len - noisy.shape[-1])) for noisy in noisy_samples])
        denoised_batched = self.denoise_single(batched, frontend_cfg=frontend_cfg, sample_rate=sample_rate)
        return [denoised[..., :audio_len] for denoised, audio_len in zip(denoised_batched, audio_lens)]

    def denoise(
        self,
        noisy_dir: str | Path,
        denoised_dir: str | Path | None = None,
        frontend_cfg: dict | None = None,
        sample_rate: int = 16000,
    ) -> list[torch.Tensor]:
        noisy_dir = Path(noisy_dir)
        denoised_dir = None if denoised_dir is None else Path(denoised_dir)

        noisy_files = [fn for fn in noisy_dir.glob("*.wav")]
        noisy_samples = [torch.tensor(librosa.load(wav_file, sr=sample_rate)[0]) for wav_file in noisy_files]
        denoised_samples = self.denoise_multiple(noisy_samples, frontend_cfg=frontend_cfg, sample_rate=sample_rate)

        if denoised_dir is not None:
            denoised_dir.mkdir(parents=True, exist_ok=True)
            for denoised, noisy_fn in zip(denoised_samples, noisy_files):
                torchaudio.save((denoised_dir / f"{noisy_fn.stem}.wav").as_posix(), denoised[None, :], sample_rate)
        return denoised_samples
