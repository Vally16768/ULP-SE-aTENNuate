from __future__ import annotations
import logging
import numpy as np
from core.data.resample_audio import resample_audio  # noqa: F401

LOG = logging.getLogger("metrics")
LOG.addHandler(logging.NullHandler())

def to_mono(x: np.ndarray) -> np.ndarray:
    """Fold to mono, flatten, float32."""
    x = np.asarray(x)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    return x.astype(np.float32).ravel()

def align(a: np.ndarray, b: np.ndarray):
    """Trim both arrays to the same (min) length."""
    n = min(len(a), len(b))
    return a[:n], b[:n]

def finite_or_default(value: float, default: float, name: str):
    """Return value if finite; otherwise log + return default."""
    if value is None or not np.isfinite(value):
        LOG.warning("Metric %s returned non-finite value (%r). Using default=%r.", name, value, default)
        return float(default)
    return float(value)

def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp scalar to [lo, hi]."""
    return float(max(lo, min(hi, val)))

def nan_to_num_inplace(x: np.ndarray, nan: float = 0.0):
    """Scrub NaN/Inf from waveforms (optional pre-step)."""
    if isinstance(x, np.ndarray):
        np.nan_to_num(x, copy=False, nan=nan, posinf=1e9, neginf=-1e9)
    return x
