import argparse
from pathlib import Path

import torch

from attenuate.model import aTENNuate


def main():
    ap = argparse.ArgumentParser(description="Exportă aTENNuate la ONNX.")
    ap.add_argument("--checkpoint", required=True,
                    help="Checkpoint .pt (FP32 sau cuantizat).")
    ap.add_argument("--out", required=True,
                    help="Fișier .onnx de ieșire.")
    ap.add_argument("--sample-len", type=int, default=16000,
                    help="Lungime exemplu dummy pentru export.")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    model = aTENNuate()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Tipul tensoului dummy: dacă parametrii sunt half, folosim half
    dtype = torch.float16 if any(p.dtype == torch.float16 for p in model.parameters()) else torch.float32
    dummy = torch.randn(1, 1, args.sample_len, dtype=dtype)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["audio_in"],
        output_names=["audio_out"],
        opset_version=args.opset,
        dynamic_axes={"audio_in": {2: "time"}, "audio_out": {2: "time"}},
    )
    print(f"Exported ONNX -> {out_path}")


if __name__ == "__main__":
    main()
