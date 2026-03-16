#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run non-intrusive metrics (DNSMOS + optional NISQA) over a dir of enhanced wavs.
"""
import argparse
import csv
import glob
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.dnsmos import dnsmos_wav

try:
    from nisqa import load_nisqa, nisqa_file
except ImportError:
    load_nisqa = None
    nisqa_file = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhanced_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument(
        "--nisqa-ckpt",
        default=None,
        help="Optional path to a NISQA checkpoint. If omitted, only DNSMOS is computed.",
    )
    args = ap.parse_args()

    wavs = sorted(glob.glob(os.path.join(args.enhanced_dir, "*.wav")))
    if not wavs:
        raise SystemExit(f"No wavs in: {args.enhanced_dir}")

    model = None
    use_nisqa = bool(args.nisqa_ckpt)
    if use_nisqa:
        if load_nisqa is None or nisqa_file is None:
            raise ImportError(
                "The 'nisqa' package is not installed, but --nisqa-ckpt was provided."
            )
        model = load_nisqa(args.nisqa_ckpt)

    rows = [("file", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr", "nisqa_mos")]
    for w in wavs:
        dns = dnsmos_wav(w)
        mos = float(nisqa_file(model, w)) if use_nisqa else ""
        rows.append(
            (
                os.path.basename(w),
                float(dns["mos_sig"]),
                float(dns["mos_bak"]),
                float(dns["mos_ovr"]),
                mos,
            )
        )

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"✔ Wrote {len(rows)-1} rows → {args.out_csv}")

if __name__ == "__main__":
    main()
