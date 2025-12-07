from __future__ import annotations
import numpy as np

_EPS = 1e-12

def _snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    c = clean.astype(np.float64).ravel()
    t = test.astype(np.float64).ravel()
    n = min(len(c), len(t))
    if n == 0:
        raise ValueError("Empty input")
    c = c[:n]
    t = t[:n]
    nrg = float(np.sum(c * c) + _EPS)
    err = c - t
    den = float(np.sum(err * err) + _EPS)
    return 10.0 * np.log10(nrg / den)

def snr_noisy(clean: np.ndarray, noisy: np.ndarray) -> float:
    return _snr_db(clean, noisy)

def snr_enhanced(clean: np.ndarray, enhanced: np.ndarray) -> float:
    return _snr_db(clean, enhanced)
