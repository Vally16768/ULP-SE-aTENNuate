import argparse
from pathlib import Path

import torch

from attenuate.model import aTENNuate


def quantize_uniform_state_dict(state_dict, num_bits: int):
    """
    Quantizare uniformă pe greutăți (per-tensor).
    32: no-op
    16: conversie la float16
    8/4/2: valori cuantizate la 2^bits nivele, păstrate ca float32.
    """
    if num_bits == 32:
        return state_dict

    new_sd = {}
    levels = 2 ** num_bits

    for k, v in state_dict.items():
        if not v.is_floating_point():
            new_sd[k] = v
            continue

        if num_bits == 16:
            new_sd[k] = v.half()
            continue

        # 8/4/2: cuantizare uniformă
        v_fp32 = v.float()
        min_val = v_fp32.min()
        max_val = v_fp32.max()
        if (max_val - min_val) < 1e-8:
            new_sd[k] = v_fp32
            continue
        scale = (max_val - min_val) / (levels - 1)
        q = torch.round((v_fp32 - min_val) / scale)
        v_q = min_val + q * scale
        new_sd[k] = v_q

    return new_sd


def main():
    ap = argparse.ArgumentParser(description="Generează checkpoint-uri cuantizate 32/16/8/4/2 biți.")
    ap.add_argument("--base-checkpoint", required=True,
                    help="Checkpoint .pt FP32 (state_dict) antrenat.")
    ap.add_argument("--out-dir", type=str, default="checkpoints_quantized",
                    help="Director unde se salvează checkpoint-urile cuantizate.")
    ap.add_argument("--bits", type=int, nargs="+", default=[32, 16, 8, 4, 2],
                    help="Lista de nivele de cuantizare.")
    args = ap.parse_args()

    base_ckpt = Path(args.base_checkpoint)
    if not base_ckpt.exists():
        raise FileNotFoundError(base_ckpt)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_state = torch.load(base_ckpt, map_location="cpu")

    for b in args.bits:
        print(f"Quantizing to {b} bits...")
        sd_q = quantize_uniform_state_dict(base_state, b)

        # validare rapidă: putem încărca într-un model
        model = aTENNuate()
        missing, unexpected = model.load_state_dict(sd_q, strict=False)
        if missing or unexpected:
            print(f"  [WARN] missing={missing}, unexpected={unexpected}")

        out_path = out_dir / f"atennuate_{b}bit.pt"
        torch.save(sd_q, out_path)
        print(f"  Saved -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
