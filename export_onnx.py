import argparse
from pathlib import Path

import torch

from attenuate.model import aTENNuate


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Calea către checkpoint-ul modelului (fp32 sau quantizat, .pt).",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Calea de ieșire pentru fișierul ONNX.",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=16000,
        help="Lungimea de secvență (T) pentru inputul dummy (ex: 16000 sample-uri).",
    )
    ap.add_argument(
        "--opset",
        type=int,
        default=18,
        help="Versiunea ONNX opset (recomandat >=18 pentru PyTorch 2.x).",
    )
    ap.add_argument(
        "--use-cuda",
        action="store_true",
        help="Dacă e setat, încarcă modelul pe CUDA pentru export (de obicei nu e necesar).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"[export] Using device: {device}")
    print(f"[export] Loading checkpoint: {ckpt_path}")

    # 1. Instanțiem modelul
    model = aTENNuate()
    state = torch.load(ckpt_path, map_location="cpu")

    # Dacă checkpoint-ul are cheia 'model' sau alt sub-dict, adaptezi aici:
    if isinstance(state, dict) and "model" in state and not any(
        k.startswith("0.") for k in state["model"].keys()
    ):
        print("[export] Detected 'model' key in checkpoint, using state['model']")
        state = state["model"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 2. Input dummy
    # Modelul tău primește (B, C, T) = (1, 1, seq_len)
    seq_len = args.seq_len
    print(f"[export] Using dummy input with seq_len={seq_len}")
    dummy_input = torch.zeros(1, 1, seq_len, dtype=torch.float32, device=device)

    # 3. Export ONNX (folosim vechiul tracer, dynamo=False)
    print(f"[export] Exporting to ONNX -> {out_path} (opset={args.opset})")

    torch.onnx.export(
        model,
        dummy_input,
        out_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["noisy"],
        output_names=["denoised"],
        # folosim exporter-ul vechi; evităm torch.export/tensor dinamic + onnxscript probleme
        dynamic_axes=None,
        verbose=False,
        dynamo=False,
    )

    print("[export] ONNX export finished successfully.")
    print(f"[export] Saved -> {out_path}")


if __name__ == "__main__":
    main()
