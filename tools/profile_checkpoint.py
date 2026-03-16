from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attenuate.runtime_profile import profile_checkpoint, save_profile_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a trained checkpoint for deployment-oriented MAC/memory estimates.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-seconds", type=float, default=1.0)
    parser.add_argument("--model-kind", default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = profile_checkpoint(
        args.checkpoint,
        device=args.device,
        input_seconds=args.input_seconds,
        model_kind=args.model_kind,
        sample_rate=args.sample_rate,
    )
    save_profile_json(profile, args.out_json)
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
