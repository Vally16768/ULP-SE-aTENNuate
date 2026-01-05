# attenuate/losses.py

from typing import Iterable, Sequence

import torch
import torchaudio
from torch import nn


class MultiResolutionERBSpectralLoss(nn.Module):
    """
    Aproximare pentru 'ERB spectral loss' din articol, folosind benzi mel.
    Lucrează pe tensori de formă (B, 1, T) sau (B, T).

    Pentru fiecare rezoluție (n_fft):
      - calculăm STFT -> magnitudine (B, F, T_frames)
      - proiectăm pe benzi mel (aprox. ERB): (B, F, T) -> (B, M, T)
      - facem L1 între energiile pe benzi
    Loss-ul final este media peste toate rezoluțiile.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_ffts: Sequence[int] | Iterable[int] = (256, 512, 1024),
        n_mels: int = 64,
        power: float = 1.0,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_ffts = tuple(n_ffts)
        self.n_mels = int(n_mels)
        self.power = float(power)
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2

        # Pre-calculăm fbanks pentru fiecare rezoluție și le înregistrăm ca buffers.
        # torchaudio.functional.melscale_fbanks returnează tipic forma (n_freqs, n_mels)
        # => F x M. O vom folosi direct ca (F, M) în matmul.
        fbanks = []
        for n_fft in self.n_ffts:
            fb = torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=self.f_min,
                f_max=self.f_max,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
                norm="slaney",
                mel_scale="slaney",
            )  # (F, M) în majoritatea versiunilor de torchaudio
            fbanks.append(fb)

        for i, fb in enumerate(fbanks):
            # fb: (F, M)
            self.register_buffer(f"fb_{i}", fb, persistent=False)

    def _stft_mag(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        """
        x: (B, T)
        return: |STFT(x)|^power, shape (B, F, T_frames)
        """
        hop_length = n_fft // 4
        win_length = n_fft
        window = torch.hann_window(win_length, device=x.device)
        X = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )  # (B, F, T_frames)
        mag = X.abs().pow(self.power)
        return mag

    def _project_to_mel(self, mag: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        """
        mag: (B, F, T) – magnitudine STFT
        fb:  (F, M)   – matrice mel (n_freqs, n_mels)

        Întoarcem: (B, M, T)
        """
        # (B, F, T) -> (B, T, F)
        mag_btF = mag.transpose(1, 2)
        # (B, T, F) x (F, M) -> (B, T, M)
        mel_btm = torch.matmul(mag_btF, fb)
        # (B, T, M) -> (B, M, T)
        mel_bmt = mel_btm.transpose(1, 2)
        return mel_bmt

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """
        enhanced, clean: (B, 1, T) sau (B, T)
        """
        if enhanced.shape != clean.shape:
            raise ValueError(
                f"enhanced și clean trebuie să aibă aceeași formă, "
                f"dar avem {enhanced.shape} vs {clean.shape}"
            )

        # Acceptăm (B, 1, T) sau (B, T); convertim la (B, T)
        if enhanced.dim() == 3:
            # (B, 1, T) -> (B, T)
            enhanced = enhanced.squeeze(1)
            clean = clean.squeeze(1)

        total_loss = 0.0
        num_resolutions = len(self.n_ffts)

        for i, n_fft in enumerate(self.n_ffts):
            fb: torch.Tensor = getattr(self, f"fb_{i}")  # (F, M)
            fb = fb.to(enhanced.device)

            mag_e = self._stft_mag(enhanced, n_fft)  # (B, F, T)
            mag_c = self._stft_mag(clean, n_fft)     # (B, F, T)

            # proiectăm pe benzi mel: (B, F, T) -> (B, M, T)
            mel_e = self._project_to_mel(mag_e, fb)
            mel_c = self._project_to_mel(mag_c, fb)

            loss_res = torch.mean(torch.abs(mel_e - mel_c))
            total_loss = total_loss + loss_res

        return total_loss / num_resolutions
