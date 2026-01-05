#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run non-intrusive metrics (DNSMOS + NISQA) over a dir of enhanced wavs. STRICT.
"""
import os, argparse, glob, csv

from dnsmos import dnsmos_wav
from nisqa import load_nisqa, nisqa_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhanced_dir", required=True)
    ap.add_argument("--out_csv",      required=True)
    ap.add_argument("--nisqa-ckpt",   required=True, help="Path to NISQA checkpoint")
    args = ap.parse_args()

    wavs = sorted(glob.glob(os.path.join(args.enhanced_dir, "*.wav")))
    if not wavs:
        raise SystemExit(f"No wavs in: {args.enhanced_dir}")

    model = load_nisqa(args.nisqa_ckpt)

    rows = [("file","dnsmos_sig","dnsmos_bak","dnsmos_ovr","nisqa_mos")]
    for w in wavs:
        dns = dnsmos_wav(w)
        mos = nisqa_file(model, w)
        rows.append((os.path.basename(w),
                     float(dns["mos_sig"]),
                     float(dns["mos_bak"]),
                     float(dns["mos_ovr"]),
                     float(mos)))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"✔ Wrote {len(rows)-1} rows → {args.out_csv}")

if __name__ == "__main__":
    main()
