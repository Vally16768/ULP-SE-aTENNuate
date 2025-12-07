# dnsmos/dnsmos_local.py — minimal, strict DNSMOS runner (P.835 + P.808)
from __future__ import annotations
from typing import Dict
import os
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort

# Exposed constants for adapters
SAMPLING_RATE = 16000

def _read_mono_16k(path: str, target_sr: int = SAMPLING_RATE) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(wav, np.ndarray) and wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")
    return wav.astype(np.float32, copy=False)

def _as_rank2(x: np.ndarray, length: int | None = None) -> np.ndarray:
    """
    Ensure shape [1, T]. If 'length' is given, pad/truncate to that length.
    """
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Expected mono vector; got shape {x.shape}")
    if length is not None:
        if x.shape[0] < length:
            pad = length - x.shape[0]
            x = np.pad(x, (0, pad), mode="constant")
        elif x.shape[0] > length:
            x = x[:length]
    return x[None, :]  # [1, T]

def _first_scalar(y) -> float:
    a = np.asarray(y).astype(np.float32)
    return float(a.reshape(-1)[0])

class ComputeScore:
    """
    Minimal DNSMOS driver:
    - primary_model_path: P.835 model (often outputs 3 scores: sig/bak/ovr)
    - p808_model_path:    P.808 model (overall MOS)
    """
    def __init__(self, primary_model_path: str, p808_model_path: str):
        if not os.path.exists(primary_model_path):
            raise FileNotFoundError(f"Primary P.835 ONNX missing: {primary_model_path}")
        if not os.path.exists(p808_model_path):
            raise FileNotFoundError(f"P.808 ONNX missing: {p808_model_path}")

        providers = ["CPUExecutionProvider"]  # explicit for reproducibility
        self.primary_sess = ort.InferenceSession(primary_model_path, providers=providers)
        self.p808_sess    = ort.InferenceSession(p808_model_path, providers=providers)

        # Cache input names & optional fixed lengths (if model declares them)
        self.primary_in = self.primary_sess.get_inputs()[0]
        self.p808_in    = self.p808_sess.get_inputs()[0]
        self.primary_name = self.primary_in.name
        self.p808_name    = self.p808_in.name

        # Derive fixed input length if the ONNX shape specifies it (e.g., [1, 48000])
        def _fixed_len(onnx_input) -> int | None:
            shp = onnx_input.shape
            if isinstance(shp, (list, tuple)) and len(shp) == 2:
                # shape like [1, N] or [None, N]
                n = shp[1]
                if isinstance(n, int):
                    return n
            return None

        self.primary_len = _fixed_len(self.primary_in)
        self.p808_len    = _fixed_len(self.p808_in)

    def __call__(self, fpath: str, sampling_rate: int = SAMPLING_RATE,
                 is_personalized_MOS: bool = False) -> Dict[str, float]:
        if not isinstance(fpath, str) or not os.path.exists(fpath):
            raise FileNotFoundError(f"WAV not found: {fpath}")

        # 1) Load & resample
        wav = _read_mono_16k(fpath, target_sr=SAMPLING_RATE)

        # 2) Build inputs — strictly rank-2 [1, T], pad/truncate if model requires fixed length
        x_primary = _as_rank2(wav, length=self.primary_len)
        x_p808    = _as_rank2(wav, length=self.p808_len)

        # 3) Run models
        p835_outs = self.primary_sess.run(None, {self.primary_name: x_primary})
        p808_outs = self.p808_sess.run(None,    {self.p808_name:    x_p808})

        # 4) Coerce to MOS dict
        # Many P.835 drops output [sig, bak, ovr] somewhere in first output; be robust:
        p835_vals = np.concatenate([np.asarray(o).reshape(-1) for o in p835_outs]).astype(np.float32)
        if p835_vals.size >= 3:
            mos_sig, mos_bak, mos_ovr = map(float, p835_vals[:3])
        elif p835_vals.size == 1:
            # If only one value, use p808 for overall and duplicate primary to SIG
            mos_sig = float(p835_vals[0])
            mos_bak = float(p835_vals[0])
            mos_ovr = float(np.asarray(p808_outs[0]).reshape(-1)[0])
            return {"mos_sig": mos_sig, "mos_bak": mos_bak, "mos_ovr": mos_ovr}
        else:
            # Fall back to p808 overall if primary is exotic
            mos_ovr = float(np.asarray(p808_outs[0]).reshape(-1)[0])
            return {"mos_sig": mos_ovr, "mos_bak": mos_ovr, "mos_ovr": mos_ovr}

        # If we have a separate p808 overall, prefer it for OVR if available
        try:
            p808_val = _first_scalar(p808_outs[0])
            mos_ovr = float(p808_val)
        except Exception:
            pass

        # Final range sanity (do not clamp)
        for v in (mos_sig, mos_bak, mos_ovr):
            if not np.isfinite(v) or not (0.0 <= v <= 5.5):
                raise ValueError(f"Invalid MOS value: {v}")

        return {"mos_sig": mos_sig, "mos_bak": mos_bak, "mos_ovr": mos_ovr}
