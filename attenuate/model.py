import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torchaudio
from einops.layers.torch import EinMix
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from huggingface_hub import hf_hub_download

from attenuate.audio import apply_input_frontend
from attenuate.mp_senet_lite import MPSENetLite
from attenuate.mp_senet_micro import MPSENetMicro
from attenuate.percepnet_class import PercepNetClass

# ------------------------------------------------------------
# FFT-based convolution helper (training / offline inference)
# ------------------------------------------------------------

def fft_conv(equation, input, kernel, *args):
    """
    Convoluție 1D în domeniul frecvență, folosită pentru SSM.
    input:  (B, C_in, T)
    kernel: (C_out, C_in, L) sau forme compatibile cu einsum-ul dat.
    """
    input, kernel = input.float(), kernel.float()
    args = tuple(arg.cfloat() for arg in args)
    n = input.shape[-1]

    kernel_f = torch.fft.rfft(kernel, 2 * n)
    input_f = torch.fft.rfft(input, 2 * n)
    output_f = torch.einsum(equation, input_f, kernel_f, *args)
    output = torch.fft.irfft(output_f, 2 * n)
    return output[..., :n]


# ------------------------------------------------------------
# SSM kernels (discrete IIR base) – Voice aTENNuate paper style
# ------------------------------------------------------------

def ssm_basis_kernels(A, B, log_dt, length: int):
    """
    A: shape (num_coeffs * repeat, 2)  -> (log_A_real, A_imag)
    B: shape (num_coeffs * repeat, C_in)
    log_dt: shape (num_coeffs * repeat,)
    length: T (lungimea semnalului)

    întoarce:
      K:      (num_coeffs * repeat, length)
      B_hat:  (num_coeffs * repeat, C_in)
    """
    log_A_real, A_imag = A.T  # (2, num_coeffs*repeat)
    lrange = torch.arange(length, device=A.device)
    dt = log_dt.exp()

    # dt * A  (real negativ + imaginar)
    dtA_real = -dt * F.softplus(log_A_real)
    dtA_imag = dt * A_imag

    # exp(dtA_real * n) * cos(dtA_imag * n)
    exponents = (dtA_real[:, None] * lrange).exp() * torch.cos(dtA_imag[:, None] * lrange)
    K = exponents                                   # (N, T)
    B_hat = B * dt[:, None]                         # (N, C_in)
    return K, B_hat


def opt_ssm_forward(input, K, B_hat, C):
    """
    SSM ops cu FFT + einsum, cu alegere de ordine de contracție în funcție de dimensiuni.

    input: (B, C_in, T)
    K:     (N, T)
    B_hat: (N, C_in)
    C:     (C_out, N)
    """
    batch, c_in, _ = input.shape
    c_out, coeffs = C.shape

    # Heuristic exact ca în codul original
    if (1 / c_in + 1 / c_out) > (1 / batch + 1 / coeffs):
        if c_in * c_out <= coeffs:
            kernel = torch.einsum('dn,nc,nl->dcl', C, B_hat, K)
            return fft_conv('bcl,dcl->bdl', input, kernel)
    else:
        if coeffs <= c_in:
            x = torch.einsum('bcl,nc->bnl', input, B_hat)
            x = fft_conv('bnl,nl->bnl', x, K)
            return torch.einsum('bnl,dn->bdl', x, C)

    return fft_conv('bcl,nl,nc,dn->bdl', input, K, B_hat, C)


# ------------------------------------------------------------
# SSM layer (training / offline)
# ------------------------------------------------------------

# configurăm opt_einsum (dacă e disponibil) la nivel de modul
try:
    from torch.backends import opt_einsum
    if opt_einsum.is_available():
        opt_einsum.strategy = "optimal"
except Exception:
    pass


class SSMLayer(nn.Module):
    """
    SSM diagonal + FFT convolution, așa cum este descris în aTENNuate.

    num_coeffs:  dimensiunea de bază a SSM (N)
    in_channels: canale de intrare
    out_channels: canale de ieșire
    repeat: câte „copii” de bază agregăm (repeat * num_coeffs total)
    """

    def __init__(
        self,
        num_coeffs: int,
        in_channels: int,
        out_channels: int,
        repeat: int,
    ):
        super().__init__()

        def init_parameter(mat):
            return Parameter(torch.tensor(mat, dtype=torch.float))

        def normal_parameter(fan_in, shape):
            return Parameter(torch.randn(*shape) * math.sqrt(2.0 / fan_in))

        # inițializare A, B, log_dt după codul original
        A_real = 0.5 * np.ones(num_coeffs)
        A_imag = math.pi * np.arange(num_coeffs)
        log_A_real = np.log(np.exp(A_real) - 1.0)  # inverse softplus
        B = np.ones(num_coeffs)
        A = np.stack([log_A_real, A_imag], -1)
        log_dt = np.linspace(np.log(0.001), np.log(0.1), repeat)

        A = np.tile(A, (repeat, 1))  # (num_coeffs*repeat, 2)
        B = np.tile(B[:, None], (repeat, in_channels)) / math.sqrt(in_channels)
        log_dt = np.repeat(log_dt, num_coeffs)      # (num_coeffs*repeat,)

        self.log_dt = init_parameter(log_dt)
        self.A = init_parameter(A)
        self.B = init_parameter(B)
        self.C = normal_parameter(
            num_coeffs * repeat,
            (out_channels, num_coeffs * repeat),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (B, C_in, T)
        return: (B, C_out, T)
        """
        K, B_hat = ssm_basis_kernels(self.A, self.B, self.log_dt, input.shape[-1])
        return opt_ssm_forward(input, K, B_hat, self.C)


# ------------------------------------------------------------
# LayerNorm pe feature (dimensiunea de canal)
# ------------------------------------------------------------

class LayerNormFeature(nn.Module):
    """
    Aplică LayerNorm pe dimensiunea canalelor (C) pentru un tensor (B, C, T).
    """
    def __init__(self, features: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, T, C) -> LN -> (B, C, T)
        return self.layer_norm(input.moveaxis(-1, -2)).moveaxis(-1, -2)


# ------------------------------------------------------------
# aTENNuate: encoder–neck–decoder + output blocks
# ------------------------------------------------------------

class aTENNuate(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels=None,
        num_coeffs: int = 16,
        repeat: int = 16,
        resample_factors=None,
        pre_conv: bool = True,
    ):
        """
        Implementarea completă a arhitecturii aTENNuate (base) din articol.

        in_channels:      1 (mono)
        channels:         [16, 32, 64, 96, 128, 256]
        resample_factors: [4, 4, 2, 2, 2, 2]
        """
        super().__init__()

        if channels is None:
            channels = [16, 32, 64, 96, 128, 256]
        if resample_factors is None:
            resample_factors = [4, 4, 2, 2, 2, 2]

        depth = len(channels)
        assert depth == len(resample_factors)

        self.depth = depth
        self.channels = [in_channels] + channels
        self.resample_factors = list(resample_factors)
        self.num_coeffs = num_coeffs
        self.repeat = repeat
        self.pre_conv = pre_conv

        # Encoder (down)
        self.down_ssms = nn.ModuleList([
            self.ssm_pool(c_in, c_out, r, downsample=True)
            for (c_in, c_out, r) in zip(self.channels[:-1], self.channels[1:], resample_factors)
        ])

        # Decoder (up)
        self.up_ssms = nn.ModuleList([
            self.ssm_pool(c_in, c_out, r, downsample=False)
            for (c_in, c_out, r) in zip(self.channels[1:], self.channels[:-1], resample_factors)
        ])

        # Neck: 2 blocuri la rezoluția cea mai joasă
        self.hid_ssms = nn.Sequential(
            self.ssm_block(self.channels[-1], use_activation=True),
            self.ssm_block(self.channels[-1], use_activation=True),
        )

        # Output: 2 blocuri pe 1 canal
        self.last_ssms = nn.Sequential(
            self.ssm_block(self.channels[0], use_activation=True),
            self.ssm_block(self.channels[0], use_activation=False),
        )

    # --------------------------------------------------------
    # Bloc SSM + resampling (down / up)
    # --------------------------------------------------------

    def ssm_pool(self, in_channels, out_channels, resample_factor, downsample: bool = True):
        """
        Downsample:
           x -> SSMBlock(in_channels) -> EinMix (b c (t r) -> b d t)

        Upsample:
           x -> EinMix (b c t -> b d (t r)) -> SSMBlock(out_channels)
        """
        if downsample:
            return nn.Sequential(
                self.ssm_block(in_channels, use_activation=True),
                EinMix(
                    'b c (t r) -> b d t',
                    weight_shape='c d r',
                    c=in_channels,
                    d=out_channels,
                    r=resample_factor,
                ),
            )
        else:
            return nn.Sequential(
                EinMix(
                    'b c t -> b d (t r)',
                    weight_shape='c d r',
                    c=in_channels,
                    d=out_channels,
                    r=resample_factor,
                ),
                self.ssm_block(out_channels, use_activation=True),
            )

    def ssm_block(self, channels: int, use_activation: bool = False) -> nn.Sequential:
        """
        Bloc: (opțional) depthwise Conv1d -> SSMLayer -> (LN + SiLU).
        """
        block = nn.Sequential()
        # PreConv depthwise, doar dacă avem mai mult de 1 canal și e activat global
        if channels > 1 and self.pre_conv:
            block.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=channels,
                )
            )
        block.append(
            SSMLayer(
                self.num_coeffs,
                channels,
                channels,
                self.repeat,
            )
        )
        if use_activation:
            if channels > 1:
                block.append(LayerNormFeature(channels))
            block.append(nn.SiLU())

        return block

    # --------------------------------------------------------
    # Forward & helper methods
    # --------------------------------------------------------

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input:  (B, 1, T)
        return: (B, 1, T)
        """
        x, skips = input, []

        # Encoder
        for ssm in self.down_ssms:
            skips.append(x)
            x = ssm(x)

        # Neck
        x = self.hid_ssms(x)

        # Decoder cu skip connections
        for (ssm, skip) in zip(self.up_ssms[::-1], skips[::-1]):
            # ssm = [EinMix_up, ssm_block]
            x = ssm[0](x)    # upsample
            x = x + skip     # skip connection
            x = ssm[1](x)    # SSMBlock

        # Output blocks (1 canal)
        x = self.last_ssms(x)
        return x

    # --------------------------------------------------------
    # Utility: denoise pe 1 / N sample-uri
    # --------------------------------------------------------

    def denoise_single(
        self,
        noisy: torch.Tensor,
        frontend_cfg: dict | None = None,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        noisy: (batch, length) la 16 kHz (mono)
        return: (batch, length) denoised
        """
        assert noisy.ndim == 2, "noisy input should be shaped (batch, length)"
        noisy = apply_input_frontend(noisy, frontend_cfg, sample_rate=sample_rate)
        noisy = noisy[:, None, :]  # unsqueeze channel dim -> (B, 1, T)

        # padding la multiplu de 256 (produsul factorilor de resampling)
        pad_factor = 256
        padding = (pad_factor - noisy.shape[-1] % pad_factor) % pad_factor
        noisy_padded = F.pad(noisy, (0, padding))
        denoised = self.forward(noisy_padded)

        return denoised.squeeze(1)[..., : noisy.shape[-1]]

    def denoise_multiple(self, noisy_samples, frontend_cfg: dict | None = None, sample_rate: int = 16000):
        """
        noisy_samples: list de tensori 1D (T_i).
        Returnează list de tensori 1D denoisați.
        """
        audio_lens = [noisy.shape[-1] for noisy in noisy_samples]
        max_len = max(audio_lens)
        batched = torch.stack([
            F.pad(noisy, (0, max_len - noisy.shape[-1]))
            for noisy in noisy_samples
        ])
        denoised_batched = self.denoise_single(batched, frontend_cfg=frontend_cfg, sample_rate=sample_rate)
        return [
            denoised[..., :audio_len]
            for (denoised, audio_len) in zip(denoised_batched, audio_lens)
        ]

    def denoise(self, noisy_dir, denoised_dir=None, frontend_cfg: dict | None = None, sample_rate: int = 16000):
        """
        API simplu pentru test:
        - citește .wav-uri din noisy_dir
        - denoise
        - opțional salvează rezultate în denoised_dir
        """
        noisy_dir = Path(noisy_dir)
        denoised_dir = None if denoised_dir is None else Path(denoised_dir)

        noisy_files = [fn for fn in noisy_dir.glob("*.wav")]
        noisy_samples = [torch.tensor(librosa.load(wav_file, sr=sample_rate)[0]) for wav_file in noisy_files]
        print("denoising...")
        denoised_samples = self.denoise_multiple(noisy_samples, frontend_cfg=frontend_cfg, sample_rate=sample_rate)

        if denoised_dir is not None:
            print("saving audio files...")
            for (denoised, noisy_fn) in zip(denoised_samples, noisy_files):
                torchaudio.save(denoised_dir / f"{noisy_fn.stem}.wav", denoised[None, :], sample_rate)

        return denoised_samples

    def from_pretrained(self, repo_id: str):
        """
        Încarcă weights.pt dintr-un repo HuggingFace (schema originală).
        """
        print(f"loading weights from {repo_id}...")
        model_weights_path = hf_hub_download(repo_id=repo_id, filename="weights.pt")
        state = torch.load(model_weights_path, map_location="cpu")
        self.load_state_dict(state)


def resolve_model_config(model_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(model_cfg or {})
    cfg["kind"] = str(cfg.get("kind", "atennuate")).strip().lower()
    return cfg


def build_model(model_cfg: dict[str, Any] | None = None) -> nn.Module:
    cfg = resolve_model_config(model_cfg)
    kind = cfg["kind"]

    if kind == "atennuate":
        kwargs = {
            "in_channels": int(cfg.get("in_channels", 1)),
            "channels": cfg.get("channels"),
            "num_coeffs": int(cfg.get("num_coeffs", 16)),
            "repeat": int(cfg.get("repeat", 16)),
            "resample_factors": cfg.get("resample_factors"),
            "pre_conv": bool(cfg.get("pre_conv", True)),
        }
        return aTENNuate(**kwargs)

    if kind == "mp_senet_lite":
        kwargs = {
            "in_channels": int(cfg.get("in_channels", 1)),
            "sample_rate": int(cfg.get("sample_rate", 16000)),
            "n_fft": int(cfg.get("n_fft", 512)),
            "hop_length": int(cfg.get("hop_length", 128)),
            "win_length": int(cfg.get("win_length", 512)),
            "compress_factor": float(cfg.get("compress_factor", 0.3)),
            "base_channels": int(cfg.get("base_channels", 32)),
            "bottleneck_channels": int(cfg.get("bottleneck_channels", 64)),
            "num_tf_blocks": int(cfg.get("num_tf_blocks", 2)),
            "num_heads": int(cfg.get("num_heads", 4)),
            "dense_growth": int(cfg.get("dense_growth", 16)),
            "dense_layers": int(cfg.get("dense_layers", 4)),
        }
        return MPSENetLite(**kwargs)

    if kind == "mp_senet_micro":
        kwargs = {
            "in_channels": int(cfg.get("in_channels", 1)),
            "sample_rate": int(cfg.get("sample_rate", 16000)),
            "channels": int(cfg.get("channels", 32)),
            "hidden_channels": int(cfg.get("hidden_channels", 64)),
            "num_blocks": int(cfg.get("num_blocks", 10)),
            "kernel_size": int(cfg.get("kernel_size", 5)),
            "dilation_cycle": int(cfg.get("dilation_cycle", 5)),
            "max_mask": float(cfg.get("max_mask", 1.2)),
            "residual_scale": float(cfg.get("residual_scale", 0.15)),
            "frame_len": cfg.get("frame_len"),
            "hop_len": cfg.get("hop_len"),
        }
        return MPSENetMicro(**kwargs)

    if kind == "percepnet_class":
        kwargs = {
            "in_channels": int(cfg.get("in_channels", 1)),
            "sample_rate": int(cfg.get("sample_rate", 16000)),
            "n_fft": cfg.get("n_fft"),
            "hop_length": cfg.get("hop_length"),
            "win_length": cfg.get("win_length"),
            "n_bands": int(cfg.get("n_bands", 32)),
            "hidden_size": int(cfg.get("hidden_size", 96)),
            "num_layers": int(cfg.get("num_layers", 2)),
            "mask_gain": float(cfg.get("mask_gain", 1.2)),
            "residual_gain": float(cfg.get("residual_gain", 0.10)),
            "stft_center": bool(cfg.get("stft_center", True)),
        }
        return PercepNetClass(**kwargs)

    raise ValueError(f"Unsupported model kind: {kind}")


def architecture_summary(model_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = resolve_model_config(model_cfg)
    model = build_model(cfg)
    summary = {
        "kind": cfg["kind"],
        "params": int(sum(param.numel() for param in model.parameters())),
        "sample_rate": int(cfg.get("sample_rate", 16000)),
    }
    if cfg["kind"] == "atennuate":
        summary.update(
            {
                "channels": list(model.channels[1:]),
                "resample_factors": list(model.resample_factors),
                "num_coeffs": int(model.num_coeffs),
                "repeat": int(model.repeat),
                "pre_conv": bool(model.pre_conv),
                "padding_multiple": 256,
                "mode": "offline waveform denoiser",
            }
        )
    elif cfg["kind"] == "mp_senet_lite":
        summary.update(
            {
                "n_fft": int(model.n_fft),
                "hop_length": int(model.hop_length),
                "win_length": int(model.win_length),
                "compress_factor": float(model.compress_factor),
                "base_channels": int(model.base_channels),
                "bottleneck_channels": int(model.bottleneck_channels),
                "num_tf_blocks": int(model.num_tf_blocks),
                "num_heads": int(model.num_heads),
                "dense_growth": int(model.dense_growth),
                "dense_layers": int(model.dense_layers),
                "padding_multiple": int(model.hop_length),
                "mode": "offline complex STFT denoiser",
            }
        )
    elif cfg["kind"] == "mp_senet_micro":
        summary.update(
            {
                "channels": int(model.channels),
                "hidden_channels": int(model.hidden_channels),
                "num_blocks": int(model.num_blocks),
                "kernel_size": int(model.kernel_size),
                "dilation_cycle": int(model.dilation_cycle),
                "frame_len": int(model.frame_len),
                "hop_len": int(model.hop_len),
                "receptive_field": int(model.receptive_field),
                "padding_multiple": 1,
                "mode": "causal streaming waveform micro denoiser",
            }
        )
    elif cfg["kind"] == "percepnet_class":
        summary.update(
            {
                "n_fft": int(model.n_fft),
                "hop_length": int(model.hop_length),
                "win_length": int(model.win_length),
                "n_bands": int(model.n_bands),
                "hidden_size": int(model.hidden_size),
                "num_layers": int(model.num_layers),
                "stft_center": bool(model.stft_center),
                "padding_multiple": int(model.hop_length),
                "mode": "hybrid ERB-GRU mask denoiser",
            }
        )
    return summary


__all__ = [
    "aTENNuate",
    "SSMLayer",
    "LayerNormFeature",
    "MPSENetLite",
    "MPSENetMicro",
    "PercepNetClass",
    "build_model",
    "resolve_model_config",
    "architecture_summary",
]
