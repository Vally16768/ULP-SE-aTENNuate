from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from einops.layers.torch import EinMix
from torch import nn

from attenuate.checkpoints import load_model_config_file, load_state_dict_file
from attenuate.model import SSMLayer, architecture_summary, build_model, resolve_model_config


ELEMENTWISE_TYPES = (
    nn.GroupNorm,
    nn.LayerNorm,
    nn.SiLU,
    nn.PReLU,
    nn.GELU,
    nn.Sigmoid,
    nn.Tanh,
    nn.ReLU,
    nn.LeakyReLU,
)

RUNTIME_CODE_BYTES = {
    "atennuate": 98_304,
    "mp_senet_lite": 131_072,
    "mp_senet_micro": 81_920,
    "percepnet_class": 98_304,
}


def _tensor_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(item) for item in obj)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(item) for item in obj.values())
    return 0


def _output_shape(output: Any) -> tuple[int, ...] | None:
    if isinstance(output, torch.Tensor):
        return tuple(int(dim) for dim in output.shape)
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return tuple(int(dim) for dim in output[0].shape)
    return None


class RuntimeProfiler:
    def __init__(self) -> None:
        self.peak_activation_bytes = 0
        self.workspace_bytes = 0
        self.bucket_ops: dict[str, float] = defaultdict(float)

    def _record_bytes(self, inputs: Any, output: Any) -> None:
        in_bytes = _tensor_bytes(inputs)
        out_bytes = _tensor_bytes(output)
        self.peak_activation_bytes = max(self.peak_activation_bytes, out_bytes)
        self.workspace_bytes = max(self.workspace_bytes, in_bytes + out_bytes)

    def hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        self._record_bytes(inputs, output)
        if isinstance(module, nn.Conv1d):
            out = output if isinstance(output, torch.Tensor) else output[0]
            batch, out_channels, out_len = out.shape
            kernel = module.kernel_size[0]
            macs = batch * out_channels * out_len * (module.in_channels // module.groups) * kernel
            self.bucket_ops["conv_int8"] += float(macs)
            return

        if isinstance(module, nn.Conv2d):
            out = output if isinstance(output, torch.Tensor) else output[0]
            batch, out_channels, out_h, out_w = out.shape
            kernel = module.kernel_size[0] * module.kernel_size[1]
            macs = batch * out_channels * out_h * out_w * (module.in_channels // module.groups) * kernel
            self.bucket_ops["conv_int8"] += float(macs)
            return

        if isinstance(module, nn.ConvTranspose2d):
            out = output if isinstance(output, torch.Tensor) else output[0]
            batch, out_channels, out_h, out_w = out.shape
            kernel = module.kernel_size[0] * module.kernel_size[1]
            macs = batch * out_channels * out_h * out_w * (module.in_channels // module.groups) * kernel
            self.bucket_ops["conv_int8"] += float(macs)
            return

        if isinstance(module, nn.Linear):
            out = output if isinstance(output, torch.Tensor) else output[0]
            macs = out.numel() * module.in_features
            self.bucket_ops["matmul_attention"] += float(macs)
            return

        if isinstance(module, nn.GRU):
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            if module.batch_first:
                batch, steps, _ = x.shape
            else:
                steps, batch, _ = x.shape
            directions = 2 if module.bidirectional else 1
            total = 0.0
            in_size = module.input_size
            for _layer in range(module.num_layers):
                gate = 3.0 * module.hidden_size
                total += batch * steps * directions * (gate * in_size + gate * module.hidden_size)
                in_size = module.hidden_size * directions
            self.bucket_ops["recurrent"] += float(total)
            return

        if isinstance(module, nn.MultiheadAttention):
            query = inputs[0]
            key = inputs[1]
            if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
                return
            if module.batch_first:
                batch, q_len, embed = query.shape
                k_len = key.shape[1]
            else:
                q_len, batch, embed = query.shape
                k_len = key.shape[0]
            proj = batch * (q_len + 2 * k_len) * embed * embed
            attn = 2.0 * batch * module.num_heads * q_len * k_len * (embed // module.num_heads)
            out_proj = batch * q_len * embed * embed
            self.bucket_ops["matmul_attention"] += float(proj + attn + out_proj)
            return

        if isinstance(module, SSMLayer):
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            batch, in_channels, seq_len = x.shape
            out_channels = int(module.C.shape[0])
            coeffs = int(module.C.shape[1])
            fft_cost = 5.0 * seq_len * math.log2(max(2, seq_len))
            macs = batch * (in_channels + out_channels + coeffs) * fft_cost
            self.bucket_ops["fft_conv"] += float(macs)
            return

        if isinstance(module, EinMix):
            shape = _output_shape(output)
            if shape is None or not hasattr(module, "weight"):
                return
            out_elems = math.prod(shape)
            weight = getattr(module, "weight")
            if not isinstance(weight, torch.Tensor):
                return
            contracted = max(1, int(weight.numel() / max(1, shape[1] if len(shape) > 1 else 1)))
            self.bucket_ops["conv_int8"] += float(out_elems * contracted)
            return

        if isinstance(module, ELEMENTWISE_TYPES):
            out = output if isinstance(output, torch.Tensor) else output[0]
            if isinstance(out, torch.Tensor):
                self.bucket_ops["elementwise"] += float(out.numel())


def _streaming_hints(kind: str, model: nn.Module, summary: dict[str, Any], sample_len: int) -> tuple[int, int, str, bool, bool]:
    if kind == "atennuate":
        return sample_len, sample_len, "offline", False, True
    if kind == "mp_senet_lite":
        return sample_len, int(summary.get("hop_length", getattr(model, "hop_length", 128))), "offline", False, True
    if kind == "mp_senet_micro":
        return int(summary.get("frame_len", getattr(model, "frame_len", sample_len))), int(summary.get("hop_len", getattr(model, "hop_len", sample_len))), "streaming", True, True
    if kind == "percepnet_class":
        causal_ready = not bool(summary.get("stft_center", getattr(model, "stft_center", True)))
        mode = "streaming" if causal_ready else "block_streaming"
        return int(summary.get("win_length", getattr(model, "win_length", sample_len))), int(summary.get("hop_length", getattr(model, "hop_length", sample_len))), mode, causal_ready, True
    return sample_len, sample_len, "offline", False, True


def _stft_bucket_ops(model: nn.Module, kind: str, sample_len: int) -> dict[str, float]:
    if not hasattr(model, "n_fft") or not hasattr(model, "hop_length"):
        return {}
    n_fft = int(getattr(model, "n_fft"))
    hop = int(getattr(model, "hop_length"))
    frame_len = int(getattr(model, "win_length", n_fft))
    frames = max(1, 1 + max(0, sample_len - frame_len) // max(1, hop))
    fft_ops = 5.0 * n_fft * math.log2(max(2, n_fft))
    result = {"fft_stft": float(2.0 * frames * fft_ops)}
    if kind == "percepnet_class":
        n_bands = int(getattr(model, "n_bands", 32))
        freqs = n_fft // 2 + 1
        result["frontend_dsp"] = float(2.0 * frames * freqs * n_bands)
    return result


def _io_bytes(frame_len: int) -> int:
    return int(frame_len * 2 * 2)


def profile_model(
    model: nn.Module,
    model_cfg: dict[str, Any],
    *,
    device: str = "cpu",
    input_seconds: float = 1.0,
) -> dict[str, Any]:
    cfg = resolve_model_config(model_cfg)
    kind = str(cfg["kind"])
    summary = architecture_summary(cfg)
    sample_rate = int(cfg.get("sample_rate", summary.get("sample_rate", 16000)))
    sample_len = max(1, int(round(sample_rate * input_seconds)))
    if kind == "mp_senet_micro":
        sample_len = max(int(summary.get("frame_len", sample_len)), int(summary.get("hop_len", sample_len)))
    elif kind == "percepnet_class":
        sample_len = max(int(summary.get("win_length", sample_len)), int(summary.get("hop_length", sample_len)))
    pad_multiple = int(summary.get("padding_multiple", 1))
    if pad_multiple > 1 and sample_len % pad_multiple != 0:
        sample_len = ((sample_len + pad_multiple - 1) // pad_multiple) * pad_multiple

    model = model.to(device)
    model.eval()

    profiler = RuntimeProfiler()
    hooks = []
    for module in model.modules():
        if module is model:
            continue
        hooks.append(module.register_forward_hook(profiler.hook))

    example = torch.zeros(1, 1, sample_len, device=device, dtype=torch.float32)
    with torch.no_grad():
        _ = model(example)

    for hook in hooks:
        hook.remove()

    for bucket, ops in _stft_bucket_ops(model, kind, sample_len).items():
        profiler.bucket_ops[bucket] += float(ops)

    frame_len, hop_len, streaming_mode, causal_ready, supports_block_inference = _streaming_hints(kind, model, summary, sample_len)
    duration_s = sample_len / sample_rate
    op_buckets_per_second = {
        key: float(value / duration_s)
        for key, value in profiler.bucket_ops.items()
        if value > 0
    }
    total_macs_per_second = float(sum(op_buckets_per_second.values()))
    mac_per_hop = float(total_macs_per_second * hop_len / sample_rate)
    num_params = int(sum(param.numel() for param in model.parameters()))

    return {
        "name": f"{kind}_{sample_rate // 1000}k",
        "family": kind,
        "sample_rate": sample_rate,
        "frame_len": int(frame_len),
        "hop_len": int(hop_len),
        "streaming_mode": streaming_mode,
        "causal_ready": bool(causal_ready),
        "supports_block_inference": bool(supports_block_inference),
        "num_params": num_params,
        "weight_bytes_fp32": int(num_params * 4),
        "weight_bytes_int8": int(num_params),
        "runtime_code_bytes": int(RUNTIME_CODE_BYTES.get(kind, 98_304)),
        "activation_peak_bytes": int(profiler.peak_activation_bytes),
        "workspace_bytes": int(profiler.workspace_bytes),
        "io_bytes": int(_io_bytes(frame_len)),
        "op_buckets_per_second": op_buckets_per_second,
        "accelerator_friendly_buckets": [bucket for bucket in ("conv_int8", "recurrent", "matmul_attention") if op_buckets_per_second.get(bucket, 0.0) > 0],
        "mac_per_second": total_macs_per_second,
        "mac_per_hop": mac_per_hop,
        "algorithmic_latency_ms": float(1000.0 * frame_len / sample_rate),
        "profile_seconds": float(duration_s),
        "notes": f"Measured from trained checkpoint using dummy inference on {duration_s:.3f}s of audio.",
    }


def profile_checkpoint(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    input_seconds: float = 1.0,
    model_kind: str | None = None,
    sample_rate: int | None = None,
) -> dict[str, Any]:
    ckpt_path = Path(checkpoint)
    model_cfg = load_model_config_file(ckpt_path, fallback={"kind": model_kind or "atennuate"})
    if model_kind is not None:
        model_cfg["kind"] = model_kind
    if sample_rate is not None:
        model_cfg["sample_rate"] = int(sample_rate)
    model = build_model(model_cfg)
    model.load_state_dict(load_state_dict_file(ckpt_path, map_location="cpu"))
    profile = profile_model(model, model_cfg, device=device, input_seconds=input_seconds)
    profile["checkpoint"] = ckpt_path.as_posix()
    profile["model_config"] = model_cfg
    return profile


def save_profile_json(profile: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
