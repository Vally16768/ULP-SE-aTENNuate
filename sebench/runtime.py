from __future__ import annotations

import torch


def require_cuda_device(device: str | None) -> str:
    requested = (device or "cuda").strip().lower()
    if not requested:
        requested = "cuda"
    if not requested.startswith("cuda"):
        raise ValueError(
            f"GPU-only workflow: expected --device cuda or cuda:N, got {device!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only workflow: CUDA is not available on this machine.")
    if requested == "cuda":
        return requested
    try:
        index = int(requested.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Invalid CUDA device {device!r}. Use --device cuda or cuda:N."
        ) from exc
    device_count = torch.cuda.device_count()
    if index < 0 or index >= device_count:
        raise ValueError(
            f"Invalid CUDA device index {index}; available device count is {device_count}."
        )
    return requested
