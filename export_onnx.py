import argparse
import importlib.util
import inspect
from pathlib import Path

import torch
from torch.onnx.errors import UnsupportedOperatorError

from attenuate.checkpoints import load_model_config_file, load_state_dict_file
from attenuate.model import architecture_summary, build_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Calea catre checkpoint-ul modelului (.pt).",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Calea de iesire pentru fisierul ONNX.",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=16000,
        help="Lungimea secventei pentru inputul dummy.",
    )
    ap.add_argument(
        "--opset",
        type=int,
        default=18,
        help="Versiunea ONNX opset dorita.",
    )
    ap.add_argument(
        "--use-cuda",
        action="store_true",
        help="Daca este setat, incarca modelul pe CUDA pentru export.",
    )
    ap.add_argument(
        "--prefer-dynamo",
        action="store_true",
        help="Incearca mai intai exporter-ul nou (dynamo) daca este disponibil.",
    )
    ap.add_argument(
        "--model-kind",
        type=str,
        default=None,
        help="Suprascrie tipul modelului daca checkpoint-ul nu are metadata.",
    )
    return ap.parse_args()


def _supports_dynamo_export() -> bool:
    try:
        return "dynamo" in inspect.signature(torch.onnx.export).parameters
    except Exception:
        return False


def _has_onnxscript() -> bool:
    return importlib.util.find_spec("onnxscript") is not None


def _legacy_export(model: torch.nn.Module, dummy_input: torch.Tensor, out_path: Path, opset: int) -> None:
    torch.onnx.export(
        model,
        dummy_input,
        out_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["noisy"],
        output_names=["denoised"],
        dynamic_axes=None,
        verbose=False,
    )


def _dynamo_export(model: torch.nn.Module, dummy_input: torch.Tensor, out_path: Path, opset: int) -> None:
    torch.onnx.export(
        model,
        dummy_input,
        out_path.as_posix(),
        export_params=True,
        opset_version=max(18, opset),
        input_names=["noisy"],
        output_names=["denoised"],
        dynamic_axes=None,
        verbose=False,
        dynamo=True,
    )


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"[export] Using device: {device}")
    print(f"[export] Loading checkpoint: {ckpt_path}")

    model_cfg = load_model_config_file(ckpt_path, fallback={"kind": args.model_kind or "atennuate"})
    if args.model_kind is not None:
        model_cfg["kind"] = args.model_kind
    summary = architecture_summary(model_cfg)
    model = build_model(model_cfg)
    model.load_state_dict(load_state_dict_file(ckpt_path, map_location="cpu"))
    model.to(device)
    model.eval()

    seq_len = args.seq_len
    padding_multiple = int(summary.get("padding_multiple", 1))
    if padding_multiple > 1 and seq_len % padding_multiple != 0:
        padded = ((seq_len + padding_multiple - 1) // padding_multiple) * padding_multiple
        print(f"[export] seq_len={seq_len} is not divisible by {padding_multiple}; using padded seq_len={padded}")
        seq_len = padded

    opset = args.opset
    legacy_opset = opset
    if legacy_opset > 17:
        print(f"[export] opset {legacy_opset} not supported by legacy exporter in this environment; using opset=17")
        legacy_opset = 17
    dummy_input = torch.zeros(1, 1, seq_len, dtype=torch.float32, device=device)
    print(f"[export] Using dummy input with seq_len={seq_len}")

    modern_error = None
    if args.prefer_dynamo and _supports_dynamo_export() and _has_onnxscript():
        print(f"[export] Trying dynamo exporter -> {out_path} (opset={max(18, opset)})")
        try:
            _dynamo_export(model, dummy_input, out_path, opset)
            print("[export] ONNX export finished successfully with dynamo exporter.")
            print(f"[export] Saved -> {out_path}")
            return
        except Exception as exc:  # noqa: BLE001
            modern_error = exc
            print(f"[export] Dynamo exporter failed; falling back to legacy exporter: {exc}")

    print(f"[export] Exporting with legacy exporter -> {out_path} (opset={legacy_opset})")
    try:
        _legacy_export(model, dummy_input, out_path, legacy_opset)
    except UnsupportedOperatorError as exc:
        modern_note = ""
        if modern_error is not None:
            modern_note = f" Dynamo exporter also failed earlier: {modern_error}"
        raise SystemExit(
            "ONNX export failed because the legacy exporter in this PyTorch build does not support "
            "FFT operators used by aTENNuate (for example aten::fft_rfft)."
            f"{modern_note} "
            "Use TorchScript export or install a newer ONNX exporter stack with FFT coverage."
        ) from exc

    print("[export] ONNX export finished successfully.")
    print(f"[export] Saved -> {out_path}")


if __name__ == "__main__":
    main()
