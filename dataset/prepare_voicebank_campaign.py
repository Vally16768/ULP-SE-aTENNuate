#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sebench.splits import build_voicebank_campaign_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create speaker-disjoint VoiceBank+DEMAND campaign splits.")
    parser.add_argument(
        "--train-csv",
        default="/mnt/ldm/ULP-SE-aTENNuate/dataset/voicebank-demand/16k/train.csv",
        help="Input VoiceBank+DEMAND train.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/ldm/ULP-SE-aTENNuate/dataset/voicebank-demand/16k/campaign",
        help="Directory where train_fit/val_pool/val_rank/val_select will be written.",
    )
    parser.add_argument(
        "--val-speakers",
        nargs="+",
        default=["p239", "p286", "p244", "p270"],
        help="Speaker ids reserved for the internal validation pool.",
    )
    parser.add_argument("--rank-count", type=int, default=128, help="Number of fixed examples in val_rank.csv.")
    args = parser.parse_args()

    manifests = build_voicebank_campaign_splits(
        train_csv=args.train_csv,
        output_dir=args.out_dir,
        val_speakers=tuple(args.val_speakers),
        rank_count=args.rank_count,
    )
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
