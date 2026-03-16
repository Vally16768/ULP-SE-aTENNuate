from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from attenuate.audio import snr_db


def hz_to_erb(hz: torch.Tensor | float) -> torch.Tensor:
    hz_t = hz if isinstance(hz, torch.Tensor) else torch.tensor(float(hz))
    return 21.4 * torch.log10(1.0 + 0.00437 * hz_t)


def erb_to_hz(erb: torch.Tensor | float) -> torch.Tensor:
    erb_t = erb if isinstance(erb, torch.Tensor) else torch.tensor(float(erb))
    return (torch.pow(torch.tensor(10.0, dtype=erb_t.dtype), erb_t / 21.4) - 1.0) / 0.00437


def build_erb_filterbank(
    n_fft: int,
    sample_rate: int,
    n_bands: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    f_max = sample_rate / 2 if f_max is None else f_max
    freqs = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    erb_min = hz_to_erb(float(f_min))
    erb_max = hz_to_erb(float(f_max))
    erb_points = torch.linspace(float(erb_min), float(erb_max), n_bands + 2)
    hz_points = erb_to_hz(erb_points)

    fb = torch.zeros(n_fft // 2 + 1, n_bands)
    centers = []
    for idx in range(n_bands):
        left = hz_points[idx]
        center = hz_points[idx + 1]
        right = hz_points[idx + 2]
        up = (freqs - left) / max(float(center - left), 1e-6)
        down = (right - freqs) / max(float(right - center), 1e-6)
        tri = torch.minimum(up, down).clamp_min(0.0)
        tri = tri / tri.sum().clamp_min(1e-6)
        fb[:, idx] = tri
        centers.append(center)
    return fb, torch.stack(centers)


def critical_band_weights(center_hz: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0:
        return torch.ones_like(center_hz)
    log_center = torch.log(center_hz.clamp_min(50.0))
    focus = torch.log(torch.tensor(1800.0, dtype=center_hz.dtype))
    weights = 1.0 + strength * torch.exp(-0.5 * ((log_center - focus) / 0.55) ** 2)
    return weights / weights.mean().clamp_min(1e-6)


class ERBFeatureProjector(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int | None = None,
        win_length: int | None = None,
        n_bands: int = 64,
        power: float = 1.0,
        log_features: bool = True,
        f_min: float = 0.0,
        f_max: float | None = None,
        band_emphasis_strength: float = 0.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length or (n_fft // 4))
        self.win_length = int(win_length or n_fft)
        self.power = float(power)
        self.log_features = bool(log_features)
        fb, centers = build_erb_filterbank(
            n_fft=self.n_fft,
            sample_rate=self.sample_rate,
            n_bands=n_bands,
            f_min=f_min,
            f_max=f_max,
        )
        weights = critical_band_weights(centers, band_emphasis_strength)
        self.register_buffer("filterbank", fb, persistent=False)
        self.register_buffer("band_weights", weights, persistent=False)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        window = torch.hann_window(self.win_length, device=x.device)
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

    def magnitude(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.stft(x)
        return spec.abs().pow(self.power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = self.magnitude(x)
        erb = torch.einsum("bft,fm->bmt", mag, self.filterbank.to(mag.device))
        erb = erb * self.band_weights.to(erb.device).view(1, -1, 1)
        if self.log_features:
            erb = torch.log1p(erb)
        return erb


class MultiResolutionERBSpectralLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_ffts: Sequence[int] | Iterable[int] = (256, 512, 1024),
        n_bands: int = 64,
        power: float = 1.0,
        f_min: float = 0.0,
        f_max: float | None = None,
        band_emphasis_strength: float = 0.0,
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleList(
            [
                ERBFeatureProjector(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    n_bands=n_bands,
                    power=power,
                    log_features=False,
                    f_min=f_min,
                    f_max=f_max,
                    band_emphasis_strength=band_emphasis_strength,
                )
                for n_fft in n_ffts
            ]
        )

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if enhanced.shape != clean.shape:
            raise ValueError(f"enhanced and clean must match, got {enhanced.shape} vs {clean.shape}")
        total = enhanced.new_tensor(0.0)
        for projector in self.projectors:
            erb_e = projector(enhanced)
            erb_c = projector(clean)
            total = total + torch.mean(torch.abs(erb_e - erb_c))
        return total / len(self.projectors)


class TorchMRSTFTLoss(nn.Module):
    def __init__(
        self,
        n_ffts: Sequence[int] | Iterable[int] = (256, 512, 1024),
        spectral_convergence_weight: float = 1.0,
        log_mag_weight: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.n_ffts = tuple(int(v) for v in n_ffts)
        self.spectral_convergence_weight = float(spectral_convergence_weight)
        self.log_mag_weight = float(log_mag_weight)
        self.eps = float(eps)

    def _mag(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        window = torch.hann_window(n_fft, device=x.device)
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        return spec.abs().clamp_min(self.eps)

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        total = enhanced.new_tensor(0.0)
        for n_fft in self.n_ffts:
            mag_e = self._mag(enhanced, n_fft)
            mag_c = self._mag(clean, n_fft)
            diff = mag_c - mag_e
            sc = torch.linalg.vector_norm(diff, dim=(1, 2)) / torch.linalg.vector_norm(mag_c, dim=(1, 2)).clamp_min(self.eps)
            log_mag = torch.mean(torch.abs(torch.log(mag_c) - torch.log(mag_e)), dim=(1, 2))
            total = total + self.spectral_convergence_weight * sc.mean() + self.log_mag_weight * log_mag.mean()
        return total / len(self.n_ffts)


class ComplexSTFTLoss(nn.Module):
    def __init__(self, n_ffts: Sequence[int] | Iterable[int] = (256, 512, 1024)) -> None:
        super().__init__()
        self.n_ffts = tuple(int(v) for v in n_ffts)

    def _stft(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        window = torch.hann_window(n_fft, device=x.device)
        return torch.stft(
            x,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        total = enhanced.new_tensor(0.0)
        for n_fft in self.n_ffts:
            spec_e = self._stft(enhanced, n_fft)
            spec_c = self._stft(clean, n_fft)
            total = total + (spec_e.real - spec_c.real).abs().mean() + (spec_e.imag - spec_c.imag).abs().mean()
        return total / len(self.n_ffts)


class SISDRLoss(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if enhanced.dim() == 3:
            enhanced = enhanced.squeeze(1)
            clean = clean.squeeze(1)
        clean_zm = clean - clean.mean(dim=-1, keepdim=True)
        enhanced_zm = enhanced - enhanced.mean(dim=-1, keepdim=True)
        alpha = torch.sum(enhanced_zm * clean_zm, dim=-1, keepdim=True) / torch.sum(clean_zm * clean_zm, dim=-1, keepdim=True).clamp_min(self.eps)
        s_target = alpha * clean_zm
        e_noise = enhanced_zm - s_target
        ratio = torch.sum(s_target * s_target, dim=-1) / torch.sum(e_noise * e_noise, dim=-1).clamp_min(self.eps)
        return -10.0 * torch.log10(ratio.clamp_min(self.eps)).mean()


class HighSNRPreservationLoss(nn.Module):
    def __init__(self, threshold_db: float = 15.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.threshold_db = float(threshold_db)
        self.eps = float(eps)

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        if enhanced.dim() == 3:
            enhanced = enhanced.squeeze(1)
            clean = clean.squeeze(1)
            noisy = noisy.squeeze(1)
        input_snr = snr_db(clean, noisy, eps=self.eps)
        mask = (input_snr >= self.threshold_db).float().view(-1, 1)
        if mask.sum() == 0:
            return enhanced.new_tensor(0.0)
        return (torch.abs(enhanced - noisy) * mask).sum() / mask.sum().clamp_min(1.0) / enhanced.shape[-1]


class PerceptualObjective(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        wave_beta: float = 0.5,
        erb_weight: float = 0.0,
        mrstft_weight: float = 0.0,
        complex_weight: float = 0.0,
        sisdr_weight: float = 0.0,
        high_snr_weight: float = 0.0,
        band_emphasis_strength: float = 0.0,
        high_snr_threshold_db: float = 15.0,
    ) -> None:
        super().__init__()
        self.wave_loss = nn.SmoothL1Loss(beta=wave_beta)
        self.erb_weight = float(erb_weight)
        self.mrstft_weight = float(mrstft_weight)
        self.complex_weight = float(complex_weight)
        self.sisdr_weight = float(sisdr_weight)
        self.high_snr_weight = float(high_snr_weight)

        self.erb_loss = (
            MultiResolutionERBSpectralLoss(sample_rate=sample_rate, band_emphasis_strength=band_emphasis_strength)
            if self.erb_weight > 0
            else None
        )
        self.mrstft_loss = TorchMRSTFTLoss() if self.mrstft_weight > 0 else None
        self.complex_loss = ComplexSTFTLoss() if self.complex_weight > 0 else None
        self.sisdr_loss = SISDRLoss() if self.sisdr_weight > 0 else None
        self.high_snr_loss = HighSNRPreservationLoss(threshold_db=high_snr_threshold_db) if self.high_snr_weight > 0 else None

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = self.wave_loss(enhanced, clean)
        components = {"wave": float(total.detach().item())}

        if self.erb_loss is not None:
            erb_val = self.erb_loss(enhanced, clean)
            total = total + self.erb_weight * erb_val
            components["erb"] = float(erb_val.detach().item())

        if self.mrstft_loss is not None:
            mr_val = self.mrstft_loss(enhanced, clean)
            total = total + self.mrstft_weight * mr_val
            components["mrstft"] = float(mr_val.detach().item())

        if self.complex_loss is not None:
            complex_val = self.complex_loss(enhanced, clean)
            total = total + self.complex_weight * complex_val
            components["complex"] = float(complex_val.detach().item())

        if self.sisdr_loss is not None:
            sisdr_val = self.sisdr_loss(enhanced, clean)
            total = total + self.sisdr_weight * sisdr_val
            components["sisdr"] = float(sisdr_val.detach().item())

        if self.high_snr_loss is not None:
            preserve_val = self.high_snr_loss(enhanced, clean, noisy)
            total = total + self.high_snr_weight * preserve_val
            components["high_snr"] = float(preserve_val.detach().item())

        components["total"] = float(total.detach().item())
        return total, components


class FrozenPESQSurrogateLoss(nn.Module):
    def __init__(self, surrogate: nn.Module, max_score: float = 4.5) -> None:
        super().__init__()
        self.surrogate = surrogate.eval()
        self.max_score = float(max_score)
        for param in self.surrogate.parameters():
            param.requires_grad_(False)

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        pred = self.surrogate(clean, enhanced)
        return (self.max_score - pred).mean()
