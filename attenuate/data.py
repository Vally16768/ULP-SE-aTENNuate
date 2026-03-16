from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn import functional as F

from attenuate.audio import clamp_audio, crop_or_pad_pair, db_to_amplitude, load_mono_audio, speaker_id_from_stem


@dataclass(frozen=True)
class PairRecord:
    noisy: Path
    clean: Path
    speaker_id: str
    utterance_id: str


def load_manifest_records(csv_path: str | Path) -> list[PairRecord]:
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "noisy" not in (reader.fieldnames or ()) or "clean" not in (reader.fieldnames or ()):
            raise ValueError(f"{csv_path} must contain columns: noisy, clean")
        records: list[PairRecord] = []
        for row in reader:
            clean = Path(row["clean"])
            noisy = Path(row["noisy"])
            utterance_id = clean.stem
            records.append(
                PairRecord(
                    noisy=noisy,
                    clean=clean,
                    speaker_id=speaker_id_from_stem(utterance_id),
                    utterance_id=utterance_id,
                )
            )
    if not records:
        raise ValueError(f"No rows found in {csv_path}")
    return records


class NoisyAugmentor:
    def __init__(self, sample_rate: int = 16000, config: dict[str, Any] | None = None):
        self.sample_rate = sample_rate
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.max_ops = int(self.config.get("max_ops", 2))
        self.base_probability = float(self.config.get("probability", 0.75))
        self.ramp_ratio = float(self.config.get("ramp_ratio", 0.3))
        self.current_epoch = 0
        self.total_epochs = 1

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        self.current_epoch = max(0, int(epoch))
        self.total_epochs = max(1, int(total_epochs))

    def _ramp_probability(self) -> float:
        if not self.enabled:
            return 0.0
        warm_epochs = max(1, int(round(self.total_epochs * self.ramp_ratio)))
        ratio = min(1.0, self.current_epoch / warm_epochs)
        return self.base_probability * ratio

    def __call__(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        rng: random.Random,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled or rng.random() > self._ramp_probability():
            return noisy, clean

        ops = [
            self._apply_gain,
            self._apply_bandwidth_limit,
            self._apply_quantization,
            self._apply_clipping,
            self._apply_reverb,
        ]
        rng.shuffle(ops)
        num_ops = rng.randint(1, min(self.max_ops, len(ops)))

        noisy_aug = noisy.clone()
        clean_aug = clean.clone()
        for op in ops[:num_ops]:
            noisy_aug, clean_aug = op(noisy_aug, clean_aug, rng)

        return clamp_audio(noisy_aug), clamp_audio(clean_aug)

    def _apply_gain(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        rng: random.Random,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        db_low, db_high = self.config.get("gain_db_range", [-6.0, 6.0])
        gain = db_to_amplitude(rng.uniform(float(db_low), float(db_high)))
        return noisy * gain, clean * gain

    def _apply_bandwidth_limit(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        rng: random.Random,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_sr = int(rng.choice(self.config.get("bandwidth_sample_rates", [8000, 12000])))
        x = noisy.unsqueeze(0)
        x = torchaudio.functional.resample(x, self.sample_rate, target_sr)
        x = torchaudio.functional.resample(x, target_sr, self.sample_rate)
        x = x.squeeze(0)
        if x.shape[-1] != noisy.shape[-1]:
            x = F.pad(x, (0, max(0, noisy.shape[-1] - x.shape[-1])))
            x = x[: noisy.shape[-1]]
        return x, clean

    def _apply_quantization(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        rng: random.Random,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = rng.choice(self.config.get("quantization_modes", ["mulaw", "uniform"]))
        bits = int(rng.choice(self.config.get("quantization_bits", [8, 6, 4])))
        if mode == "mulaw":
            quantized = torchaudio.functional.mu_law_encoding(clamp_audio(noisy), 1 << bits)
            restored = torchaudio.functional.mu_law_decoding(quantized, 1 << bits)
            return restored.float(), clean

        levels = float((1 << bits) - 1)
        x = clamp_audio(noisy)
        restored = torch.round((x + 1.0) * 0.5 * levels) / levels
        restored = restored * 2.0 - 1.0
        return restored.float(), clean

    def _apply_clipping(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        rng: random.Random,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        threshold_low, threshold_high = self.config.get("clip_threshold_range", [0.35, 0.95])
        threshold = float(rng.uniform(float(threshold_low), float(threshold_high)))
        if rng.random() < 0.5:
            return torch.clamp(noisy, -threshold, threshold), clean
        drive = rng.uniform(1.25, 3.0)
        return torch.tanh(noisy * drive) / torch.tanh(torch.tensor(drive)), clean

    def _apply_reverb(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        rng: random.Random,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ir_ms_low, ir_ms_high = self.config.get("reverb_ir_ms_range", [20.0, 80.0])
        ir_len = max(8, int(self.sample_rate * rng.uniform(float(ir_ms_low), float(ir_ms_high)) / 1000.0))
        wet = float(rng.uniform(0.08, 0.35))

        t = torch.arange(ir_len, dtype=noisy.dtype)
        tau = rng.uniform(0.1, 0.5) * ir_len
        decay = torch.exp(-t / tau)
        ir = torch.randn(ir_len, dtype=noisy.dtype) * decay
        ir[0] = 1.0
        ir = ir / ir.abs().sum().clamp_min(1e-6)

        rev = F.conv1d(
            noisy.view(1, 1, -1),
            ir.flip(0).view(1, 1, -1),
            padding=ir_len - 1,
        ).view(-1)[: noisy.shape[-1]]
        mixed = (1.0 - wet) * noisy + wet * rev
        return mixed, clean


class VoiceBankDemandDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        segment_len: int = 32000,
        sample_rate: int = 16000,
        augmentor: NoisyAugmentor | None = None,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.sample_rate = sample_rate
        self.segment_len = int(segment_len)
        self.augmentor = augmentor
        self.seed = int(seed)
        self.current_epoch = 0
        self.records = load_manifest_records(self.csv_path)

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        self.current_epoch = int(epoch)
        if self.augmentor is not None:
            self.augmentor.set_epoch(epoch, total_epochs)

    def __len__(self) -> int:
        return len(self.records)

    def _rng_for_index(self, idx: int) -> random.Random:
        return random.Random(self.seed + idx + self.current_epoch * 100_003)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        rng = self._rng_for_index(idx)

        noisy, _ = load_mono_audio(record.noisy, self.sample_rate)
        clean, _ = load_mono_audio(record.clean, self.sample_rate)
        noisy, clean = crop_or_pad_pair(noisy, clean, self.segment_len, rng)

        if self.augmentor is not None:
            noisy, clean = self.augmentor(noisy, clean, rng)

        return noisy, clean
