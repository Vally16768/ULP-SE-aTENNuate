# metrics/dnsmos.py — STRICT adapter for Microsoft DNSMOS (vendored)
# - Uses explicit ONNX paths from your repo.
# - Calls ComputeScore with a *file path*, not waveform.
# - No silent fallbacks; raises with clear errors if anything is off.

from __future__ import annotations
from typing import Dict
import importlib
import os
import numpy as np

# ---------------------------------------------------------------------------
# 1) Import the vendored module: project_root/dnsmos/dnsmos_local.py
# ---------------------------------------------------------------------------
try:
    m = importlib.import_module("dnsmos.dnsmos_local")
except Exception as e:
    raise ImportError(
        "Cannot import 'dnsmos.dnsmos_local'. Ensure you have a local package "
        "'dnsmos' with dnsmos_local.py at project root."
    ) from e

# ---------------------------------------------------------------------------
# 2) Explicit model paths (your folder layout)
#    dnsmos/DNSMOS/sig_bak_ovr.onnx    -> primary P.835 model
#    dnsmos/pDNSMOS/sig_bak_ovr.onnx   -> P.808 model
#    You can override via env if you rename files.
# ---------------------------------------------------------------------------
PRIMARY_ONNX = os.environ.get("DNSMOS_PRIMARY_ONNX", "dnsmos/DNSMOS/sig_bak_ovr.onnx")
P808_ONNX    = os.environ.get("DNSMOS_P808_ONNX",    "dnsmos/pDNSMOS/sig_bak_ovr.onnx")

for path in (PRIMARY_ONNX, P808_ONNX):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DNSMOS model not found: {path}\n"
            "Set DNSMOS_PRIMARY_ONNX / DNSMOS_P808_ONNX or place the ONNX files at the paths above."
        )

# ---------------------------------------------------------------------------
# 3) Instantiate the scorer (STRICT: explicit constructor)
# ---------------------------------------------------------------------------
if not hasattr(m, "ComputeScore"):
    raise ImportError("dnsmos_local.ComputeScore not found in your dnsmos_local.py")

ComputeScore = getattr(m, "ComputeScore")
scorer = ComputeScore(primary_model_path=PRIMARY_ONNX, p808_model_path=P808_ONNX)

# Target sampling rate (most drops expose SAMPLING_RATE=16000). Default to 16000 if absent.
TARGET_SR: int = int(getattr(m, "SAMPLING_RATE", 16000))

# ---------------------------------------------------------------------------
# 4) Helpers
# ---------------------------------------------------------------------------
def _coerce_to_dict(res) -> Dict[str, float]:
    """Return {'mos_sig','mos_bak','mos_ovr'} or raise."""
    if isinstance(res, dict):
        lower = {str(k).lower(): float(v) for k, v in res.items()}
        def pick(*names):
            for n in names:
                if n in lower:
                    return lower[n]
            return None
        out = {
            "mos_sig": pick("mos_sig", "sig", "signal"),
            "mos_bak": pick("mos_bak", "bak", "background"),
            "mos_ovr": pick("mos_ovr", "ovr", "overall"),
        }
        if any(v is None for v in out.values()):
            raise RuntimeError(f"Unexpected DNSMOS dict keys: {list(res.keys())}")
        return out

    if isinstance(res, (list, tuple)) and len(res) == 3:
        a, b, c = res
        return {"mos_sig": float(a), "mos_bak": float(b), "mos_ovr": float(c)}

    raise RuntimeError(f"Unsupported DNSMOS return type: {type(res).__name__}")

def _range_check(d: Dict[str, float]) -> None:
    for k, v in d.items():
        if not np.isfinite(v) or not (0.0 <= v <= 5.5):
            raise ValueError(f"DNSMOS {k} out of expected range [0, 5.5]: {v}")

# ---------------------------------------------------------------------------
# 5) Public API — call with FILE PATH (matches your ComputeScore.__call__)
# ---------------------------------------------------------------------------
def dnsmos_wav(path: str) -> Dict[str, float]:
    """Strict DNSMOS call. Returns a dict with keys: mos_sig, mos_bak, mos_ovr."""
    if not isinstance(path, str) or not os.path.exists(path):
        raise FileNotFoundError(f"WAV not found: {path}")

    # Prefer file-based APIs if present
    if hasattr(scorer, "score_file") and callable(scorer.score_file):
        res = scorer.score_file(path)
    elif hasattr(scorer, "predict_file") and callable(scorer.predict_file):
        res = scorer.predict_file(path)
    else:
        # Your variant requires a file path to __call__(fpath, sampling_rate, is_personalized_MOS)
        fn = getattr(scorer, "__call__", None)
        if not callable(fn):
            raise RuntimeError(
                "ComputeScore exposes neither score_file/predict_file nor __call__. "
                "Cannot run DNSMOS."
            )
        # First try the explicit (fpath, sr, is_personalized_MOS)
        try:
            res = fn(path, TARGET_SR, False)
        except TypeError:
            # Some variants accept only (fpath, sr)
            try:
                res = fn(path, TARGET_SR)
            except TypeError as e:
                raise TypeError(
                    "__call__ signature not supported. Expected (fpath, sampling_rate, is_personalized_MOS) "
                    "or (fpath, sampling_rate)."
                ) from e

    out = _coerce_to_dict(res)
    _range_check(out)
    return out
