import argparse
from pathlib import Path

import torch
from attenuate.checkpoints import load_model_config_file, load_state_dict_file
from attenuate.model import architecture_summary, build_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=32000)
    ap.add_argument("--model-kind", default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    print(f"[ts-export] Using device: {device}")
    print(f"[ts-export] Loading checkpoint: {ckpt_path}")

    model_cfg = load_model_config_file(ckpt_path, fallback={"kind": args.model_kind or "atennuate"})
    if args.model_kind is not None:
        model_cfg["kind"] = args.model_kind
    summary = architecture_summary(model_cfg)
    model = build_model(model_cfg)
    model.load_state_dict(load_state_dict_file(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    seq_len = args.seq_len
    padding_multiple = int(summary.get("padding_multiple", 1))
    if padding_multiple > 1 and seq_len % padding_multiple != 0:
        padded = ((seq_len + padding_multiple - 1) // padding_multiple) * padding_multiple
        print(f"[ts-export] seq_len={seq_len} is not divisible by {padding_multiple}; using padded seq_len={padded}")
        seq_len = padded
    example = torch.zeros(1, 1, seq_len, dtype=torch.float32, device=device)

    try:
        print("[ts-export] Trying torch.jit.script first")
        scripted = torch.jit.script(model)
        scripted.save(out_path.as_posix())
        print(f"[ts-export] TorchScript saved via scripting -> {out_path}")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"[ts-export] Scripting failed, falling back to tracing: {exc}")

    print(f"[ts-export] Tracing with seq_len={seq_len} (check_trace=False)")
    traced = torch.jit.trace(model, example, check_trace=False)
    traced.save(out_path.as_posix())
    print(f"[ts-export] TorchScript saved via tracing -> {out_path}")


if __name__ == "__main__":
    main()
