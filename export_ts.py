import argparse
from pathlib import Path

import torch
from attenuate.model import aTENNuate


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=32000)
    return ap.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    print(f"[ts-export] Using device: {device}")
    print(f"[ts-export] Loading checkpoint: {ckpt_path}")

    model = aTENNuate()
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    example = torch.zeros(1, 1, args.seq_len, dtype=torch.float32, device=device)

    print(f"[ts-export] Tracing with seq_len={args.seq_len}")
    traced = torch.jit.trace(model, example)
    traced.save(out_path.as_posix())
    print(f"[ts-export] TorchScript saved -> {out_path}")


if __name__ == "__main__":
    main()
