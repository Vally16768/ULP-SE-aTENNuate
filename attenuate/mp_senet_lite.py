from __future__ import annotations

import math
from pathlib import Path

import librosa
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from attenuate.audio import apply_input_frontend


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        transpose: bool = False,
    ) -> None:
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=(0, 0),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DenseDilatedBlock2d(nn.Module):
    def __init__(self, channels: int, *, growth: int = 16, num_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = channels
        for idx in range(num_layers):
            dilation = 2**idx
            block = nn.Sequential(
                nn.Conv2d(
                    current_channels,
                    growth,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(dilation, 1),
                    dilation=(dilation, 1),
                ),
                nn.GroupNorm(_group_count(growth, preferred=4), growth),
                nn.PReLU(growth),
            )
            self.layers.append(block)
            current_channels += growth
        self.project = nn.Conv2d(current_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [x]
        for layer in self.layers:
            dense = torch.cat(states, dim=1)
            states.append(layer(dense))
        return x + self.project(torch.cat(states, dim=1))


class AxialAttentionBlock2d(nn.Module):
    def __init__(self, channels: int, *, num_heads: int = 4, ff_mult: int = 2) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(channels)
        self.time_attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.freq_norm = nn.LayerNorm(channels)
        self.freq_attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        hidden = channels * ff_mult
        self.ff_norm = nn.GroupNorm(_group_count(channels), channels)
        self.ff = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, freqs = x.shape

        time_seq = x.permute(0, 3, 2, 1).reshape(batch * freqs, frames, channels)
        time_seq = self.time_norm(time_seq)
        time_out, _ = self.time_attn(time_seq, time_seq, time_seq, need_weights=False)
        x = x + time_out.reshape(batch, freqs, frames, channels).permute(0, 3, 2, 1)

        freq_seq = x.permute(0, 2, 3, 1).reshape(batch * frames, freqs, channels)
        freq_seq = self.freq_norm(freq_seq)
        freq_out, _ = self.freq_attn(freq_seq, freq_seq, freq_seq, need_weights=False)
        x = x + freq_out.reshape(batch, frames, freqs, channels).permute(0, 3, 1, 2)

        return x + self.ff(self.ff_norm(x))


class LearnableSigmoid2d(nn.Module):
    def __init__(self, n_freqs: int, beta: float = 1.2) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(n_freqs))
        self.beta = float(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, 1, 1, -1)
        return self.beta * torch.sigmoid(alpha * x)


class MPSENetLite(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        compress_factor: float = 0.3,
        base_channels: int = 32,
        bottleneck_channels: int = 64,
        num_tf_blocks: int = 2,
        num_heads: int = 4,
        dense_growth: int = 16,
        dense_layers: int = 4,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("MPSENetLite currently supports mono input only")

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.compress_factor = float(compress_factor)
        self.base_channels = int(base_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.num_tf_blocks = int(num_tf_blocks)
        self.num_heads = int(num_heads)
        self.dense_growth = int(dense_growth)
        self.dense_layers = int(dense_layers)
        self.n_freqs = self.n_fft // 2 + 1

        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

        self.encoder_in = ConvNormAct2d(3, self.base_channels)
        self.encoder_dense = DenseDilatedBlock2d(
            self.base_channels,
            growth=self.dense_growth,
            num_layers=self.dense_layers,
        )
        self.encoder_down = ConvNormAct2d(
            self.base_channels,
            self.bottleneck_channels,
            kernel_size=(3, 3),
            stride=(1, 2),
        )

        self.bridge = nn.Sequential(
            *[
                AxialAttentionBlock2d(
                    self.bottleneck_channels,
                    num_heads=self.num_heads,
                )
                for _ in range(self.num_tf_blocks)
            ]
        )

        self.mag_dense = DenseDilatedBlock2d(
            self.bottleneck_channels,
            growth=self.dense_growth,
            num_layers=self.dense_layers,
        )
        self.mag_up = ConvNormAct2d(
            self.bottleneck_channels,
            self.base_channels,
            kernel_size=(3, 3),
            stride=(1, 2),
            transpose=True,
        )
        self.mag_mask = nn.Conv2d(self.base_channels, 1, kernel_size=1)
        self.mag_activation = LearnableSigmoid2d(self.n_freqs)

        self.phase_dense = DenseDilatedBlock2d(
            self.bottleneck_channels,
            growth=self.dense_growth,
            num_layers=self.dense_layers,
        )
        self.phase_up = ConvNormAct2d(
            self.bottleneck_channels,
            self.base_channels,
            kernel_size=(3, 3),
            stride=(1, 2),
            transpose=True,
        )
        self.phase_out = nn.Conv2d(self.base_channels, 2, kernel_size=1)

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(spec.device),
            length=length,
        )

    def _prepare_features(self, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude = spec.abs().clamp_min(1e-8)
        phase = torch.angle(spec)
        compressed = magnitude.pow(self.compress_factor)
        features = torch.stack(
            [
                compressed.transpose(1, 2),
                torch.cos(phase).transpose(1, 2),
                torch.sin(phase).transpose(1, 2),
            ],
            dim=1,
        )
        return features, compressed.transpose(1, 2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError("Expected input shaped (batch, 1, time)")
        waveform = input[:, 0, :]
        length = waveform.shape[-1]

        noisy_spec = self._stft(waveform)
        features, compressed_mag = self._prepare_features(noisy_spec)

        enc = self.encoder_in(features)
        skip = self.encoder_dense(enc)
        latent = self.encoder_down(skip)
        latent = self.bridge(latent)

        mag_latent = self.mag_dense(latent)
        mag_up = self.mag_up(mag_latent)
        mag_up = mag_up[..., : skip.shape[-2], : skip.shape[-1]] + skip
        mag_mask = self.mag_activation(self.mag_mask(mag_up)).squeeze(1)

        phase_latent = self.phase_dense(latent)
        phase_up = self.phase_up(phase_latent)
        phase_up = phase_up[..., : skip.shape[-2], : skip.shape[-1]] + skip
        phase_vec = self.phase_out(phase_up)
        phase_vec = F.normalize(phase_vec, dim=1, eps=1e-8)
        phase_hat = torch.atan2(phase_vec[:, 1], phase_vec[:, 0])

        enhanced_mag = (compressed_mag * mag_mask).clamp_min(1e-8).pow(1.0 / self.compress_factor)
        enhanced_spec = torch.polar(enhanced_mag.transpose(1, 2), phase_hat.transpose(1, 2))
        enhanced = self._istft(enhanced_spec, length=length)
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
        batched = torch.stack([
            F.pad(noisy, (0, max_len - noisy.shape[-1]))
            for noisy in noisy_samples
        ])
        denoised_batched = self.denoise_single(batched, frontend_cfg=frontend_cfg, sample_rate=sample_rate)
        return [
            denoised[..., :audio_len]
            for denoised, audio_len in zip(denoised_batched, audio_lens)
        ]

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


__all__ = ["MPSENetLite"]
