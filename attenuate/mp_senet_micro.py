from __future__ import annotations

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


class CausalDepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int = 5, dilation: int = 1, expansion: int = 2) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        hidden = channels * int(expansion)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise_in = nn.Conv1d(channels, hidden, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(hidden, preferred=4), hidden)
        self.act = nn.SiLU()
        self.pointwise_out = nn.Conv1d(hidden, channels, kernel_size=1, bias=False)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    @property
    def left_padding(self) -> int:
        return (self.kernel_size - 1) * self.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, (self.left_padding, 0))
        y = self.depthwise(y)
        y = self.pointwise_in(y)
        y = self.norm(y)
        y = self.act(y)
        y = self.pointwise_out(y)
        return x + self.res_scale.tanh() * y


class MPSENetMicro(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        sample_rate: int = 16000,
        channels: int = 32,
        hidden_channels: int = 64,
        num_blocks: int = 10,
        kernel_size: int = 5,
        dilation_cycle: int = 5,
        max_mask: float = 1.2,
        residual_scale: float = 0.15,
        frame_len: int | None = None,
        hop_len: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("MPSENetMicro currently supports mono input only")

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)
        self.dilation_cycle = int(dilation_cycle)
        self.max_mask = float(max_mask)
        self.residual_scale = float(residual_scale)
        self.frame_len = int(frame_len or max(80, self.sample_rate // 50))
        self.hop_len = int(hop_len or self.frame_len)

        self.input_proj = nn.Sequential(
            nn.Conv1d(1, self.channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(self.channels), self.channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                CausalDepthwiseSeparableBlock(
                    self.channels,
                    kernel_size=self.kernel_size,
                    dilation=2 ** (idx % max(1, self.dilation_cycle)),
                )
                for idx in range(self.num_blocks)
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv1d(self.channels, self.hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(self.hidden_channels), self.hidden_channels),
            nn.SiLU(),
            nn.Conv1d(self.hidden_channels, self.channels, kernel_size=1, bias=False),
        )
        self.mask_head = nn.Conv1d(self.channels, 1, kernel_size=1)
        self.residual_head = nn.Conv1d(self.channels, 1, kernel_size=1)

    @property
    def receptive_field(self) -> int:
        base = 1
        for block in self.blocks:
            base += block.left_padding
        return base

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError("Expected input shaped (batch, 1, time)")
        noisy = input
        x = self.input_proj(noisy)
        for block in self.blocks:
            x = block(x)
        x = x + self.bottleneck(x)
        mask = torch.sigmoid(self.mask_head(x)) * self.max_mask
        residual = torch.tanh(self.residual_head(x)) * self.residual_scale
        enhanced = noisy * mask + residual
        return enhanced

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
        batched = torch.stack([F.pad(noisy, (0, max_len - noisy.shape[-1])) for noisy in noisy_samples])
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
