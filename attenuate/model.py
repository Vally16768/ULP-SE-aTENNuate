import math
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from einops.layers.torch import EinMix
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from huggingface_hub import hf_hub_download

# ------------------------------------------------------------
# FFT-based convolution helper (training / offline inference)
# ------------------------------------------------------------

@torch.compiler.disable
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

    def denoise_single(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        noisy: (batch, length) la 16 kHz (mono)
        return: (batch, length) denoised
        """
        assert noisy.ndim == 2, "noisy input should be shaped (batch, length)"
        noisy = noisy[:, None, :]  # unsqueeze channel dim -> (B, 1, T)

        # padding la multiplu de 256 (produsul factorilor de resampling)
        pad_factor = 256
        padding = (pad_factor - noisy.shape[-1] % pad_factor) % pad_factor
        noisy_padded = F.pad(noisy, (0, padding))
        denoised = self.forward(noisy_padded)

        return denoised.squeeze(1)[..., : noisy.shape[-1]]

    def denoise_multiple(self, noisy_samples):
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
        denoised_batched = self.denoise_single(batched)
        return [
            denoised[..., :audio_len]
            for (denoised, audio_len) in zip(denoised_batched, audio_lens)
        ]

    def denoise(self, noisy_dir, denoised_dir=None):
        """
        API simplu pentru test:
        - citește .wav-uri din noisy_dir
        - denoise
        - opțional salvează rezultate în denoised_dir
        """
        noisy_dir = Path(noisy_dir)
        denoised_dir = None if denoised_dir is None else Path(denoised_dir)

        noisy_files = [fn for fn in noisy_dir.glob("*.wav")]
        noisy_samples = [torch.tensor(librosa.load(wav_file, sr=16000)[0]) for wav_file in noisy_files]
        print("denoising...")
        denoised_samples = self.denoise_multiple(noisy_samples)

        if denoised_dir is not None:
            print("saving audio files...")
            for (denoised, noisy_fn) in zip(denoised_samples, noisy_files):
                torchaudio.save(denoised_dir / f"{noisy_fn.stem}.wav", denoised[None, :], 16000)

        return denoised_samples

    def from_pretrained(self, repo_id: str):
        """
        Încarcă weights.pt dintr-un repo HuggingFace (schema originală).
        """
        print(f"loading weights from {repo_id}...")
        model_weights_path = hf_hub_download(repo_id=repo_id, filename="weights.pt")
        state = torch.load(model_weights_path, map_location="cpu")
        self.load_state_dict(state)


__all__ = ["aTENNuate", "SSMLayer", "LayerNormFeature"]
