"""
Evaluate intrusive metrics over a CSV manifest with columns:
   clean,noisy,enhanced
Writes a JSON report and exits non-zero if thresholds are violated.

Note: This script is for offline evaluation (not training).
"""
from __future__ import annotations

import csv
import json
import sys
import os
import statistics as st
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from core.data.resample_audio import resample_audio

# Import local metric modules (relative)
from .pesq import pesq_score
from .stoi import stoi_score
from .snr import delta_snr
from .sisdr import sisdr


DEFAULT_THRESH = {
    "PESQ": 3.00,
    "STOI": 0.93,
    "DELTA_SNR": 9.0,      # dB
    "SI_SDR": 10.0,        # dB
    "STREAM_LAT_MS": 40.0, # placeholder for streaming runs
}


def _load(path: str):
    x, fs = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=-1)  # fold to mono
    return x.astype(np.float32), int(fs)


def _match_sr_to(ref_sig: np.ndarray, ref_fs: int, x: np.ndarray, x_fs: int):
    if x_fs != ref_fs:
        x = resample_audio(x, x_fs, ref_fs)
    n = min(len(ref_sig), len(x))
    return ref_sig[:n], x[:n], ref_fs


def eval_pair(clean_path: str, noisy_path: str, enhanced_path: str) -> Dict[str, float]:
    clean, fs_c = _load(clean_path)
    noisy, fs_n = _load(noisy_path)
    enh, fs_e = _load(enhanced_path)

    # Normalize SR: align noisy/enhanced to clean's SR/length
    clean2, noisy2, fs_use = _match_sr_to(clean, fs_c, noisy, fs_n)
    clean3, enh2, _ = _match_sr_to(clean2, fs_use, enh, fs_e)

    # Intrusive metrics (raise on error; we want hard failure in CI)
    pesq = pesq_score(clean3, enh2, fs_use)
    stoi = stoi_score(clean3, enh2, fs_use, extended=False)
    dsnr = delta_snr(clean3, noisy2, enh2)
    sdr = sisdr(clean3, enh2)

    return {
        "PESQ": float(pesq),
        "STOI": float(stoi),
        "DELTA_SNR": float(dsnr),
        "SI_SDR": float(sdr),
    }


def _read_manifest(manifest_csv: str) -> List[Dict[str, str]]:
    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty.")
    required = {"clean", "noisy", "enhanced"}
    if not required.issubset(rows[0].keys()):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    return rows


def main(manifest_csv: str, out_json: str, thresholds=None):
    """
    rulează metricile intrusive pe manifest_csv și scrie un raport JSON în out_json.
    """
    thr: Dict[str, float] = {**DEFAULT_THRESH, **(thresholds or {})}
    rows_in = _read_manifest(manifest_csv)

    scores: List[Dict[str, float]] = []
    out_rows: List[Dict[str, Any]] = []

    for r in rows_in:
        for k in ("clean", "noisy", "enhanced"):
            if not os.path.exists(r[k]):
                raise FileNotFoundError(f"{k} file not found: {r[k]}")
        sc = eval_pair(r["clean"], r["noisy"], r["enhanced"])
        scores.append(sc)
        out_rows.append({**r, **{k: float(v) for k, v in sc.items()}})

    # agregare
    agg = {k: st.mean([s[k] for s in scores]) for k in scores[0].keys()}
    failures = [(k, agg[k], thr[k]) for k in agg if k in thr and agg[k] < thr[k]]

    report: Dict[str, Any] = {
        "thresholds": thr,
        "aggregate": {k: float(v) for k, v in agg.items()},
        "count": len(scores),
        "failures": [(k, float(v), float(t)) for (k, v, t) in failures],
        "rows": out_rows,
    }

    # JSON-friendly: convertim numpy.* în tipuri native
    def _jsonize(obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, dict):
            return {k: _jsonize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonize(v) for v in obj]
        if isinstance(obj, np.generic):     # np.float32, np.int64, etc.
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    report = _jsonize(report)

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("[ORACLE]", {k: round(v, 4) for k, v in agg.items()})
    if failures:
        print("[ORACLE][FAIL]", failures)
        sys.exit(2)


if __name__ == "__main__":
    # mod standalone: poate fi rulat direct pe un manifest existent (clean,noisy,enhanced)
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with columns: clean,noisy,enhanced")
    ap.add_argument("--out", required=True, help="JSON output file")
    ap.add_argument("--thresholds", default=None, help="JSON string or path to .json")
    args = ap.parse_args()

    thr = None
    if args.thresholds:
        if args.thresholds.strip().startswith("{"):
            thr = json.loads(args.thresholds)
        else:
            with open(args.thresholds) as f:
                thr = json.load(f)
    main(args.manifest, args.out, thr)
