from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import urllib.request
from pathlib import Path

import numpy as np

__all__ = [
    "pesq",
    "pesq_batch",
    "PesqError",
    "InvalidSampleRateError",
    "OutOfMemoryError",
    "BufferTooShortError",
    "NoUtterancesError",
]


class PesqError(RuntimeError):
    SUCCESS = 0
    UNKNOWN = -1
    INVALID_SAMPLE_RATE = -2
    OUT_OF_MEMORY_REF = -3
    OUT_OF_MEMORY_DEG = -4
    OUT_OF_MEMORY_TMP = -5
    BUFFER_TOO_SHORT = -6
    NO_UTTERANCES_DETECTED = -7

    RAISE_EXCEPTION = 0
    RETURN_VALUES = 1


class InvalidSampleRateError(PesqError):
    pass


class OutOfMemoryError(PesqError):
    pass


class BufferTooShortError(PesqError):
    pass


class NoUtterancesError(PesqError):
    pass


_ROOT = Path(__file__).resolve().parent
_BUILD_DIR = _ROOT / "_build"
_SRC_CACHE_DIR = _BUILD_DIR / "src"
_BACKEND_NAME = "pesq_backend"
_BACKEND_PATH = _BUILD_DIR / f"{_BACKEND_NAME}{'.dll' if os.name == 'nt' else '.so'}"
_WRAPPER_C = _ROOT / "_backend.c"
_PYPI_JSON_URL = "https://pypi.org/pypi/pesq/0.0.4/json"
_LOAD_LOCK = threading.Lock()
_BACKEND: ctypes._CFuncPtr | None = None

_ERROR_MESSAGES = {
    PesqError.SUCCESS: "Success",
    PesqError.UNKNOWN: "Unknown",
    PesqError.INVALID_SAMPLE_RATE: "Invalid sampling rate",
    PesqError.OUT_OF_MEMORY_REF: "Unable to allocate memory for reference buffer",
    PesqError.OUT_OF_MEMORY_DEG: "Unable to allocate memory for degraded buffer",
    PesqError.OUT_OF_MEMORY_TMP: "Unable to allocate memory for temporary buffer",
    PesqError.BUFFER_TOO_SHORT: "Buffer needs to be at least 1/4 of a second long",
    PesqError.NO_UTTERANCES_DETECTED: "No utterances detected",
}


def _candidate_compilers() -> list[list[str]]:
    env_cc = os.environ.get("PESQ_CC")
    if env_cc:
        return [[env_cc]]

    candidates: list[list[str]] = []
    zig_path = shutil.which("zig")
    if zig_path:
        candidates.append([zig_path, "cc"])

    if os.name == "nt":
        winget_glob = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        for match in winget_glob.glob("zig.zig_*/*/zig.exe"):
            candidates.append([str(match), "cc"])
            break
    else:
        for name in ("cc", "gcc", "clang"):
            path = shutil.which(name)
            if path:
                candidates.append([path])

    return candidates


def _download_sdist() -> Path:
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = _BUILD_DIR / "pesq_pypi.json"
    tar_path = _BUILD_DIR / "pesq-0.0.4.tar.gz"

    if tar_path.exists():
        return tar_path

    with urllib.request.urlopen(_PYPI_JSON_URL, timeout=30) as resp:
        meta = json.load(resp)

    sdist_url = None
    for url_info in meta.get("urls", []):
        if url_info.get("packagetype") == "sdist":
            sdist_url = url_info.get("url")
            break
    if not sdist_url:
        raise RuntimeError("Could not locate the PESQ source distribution on PyPI.")

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    urllib.request.urlretrieve(sdist_url, tar_path)
    return tar_path


def _extract_source() -> Path:
    root = _SRC_CACHE_DIR / "pesq-0.0.4"
    pesq_src = root / "pesq"
    if pesq_src.exists():
        return root

    tar_path = _download_sdist()
    _SRC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(_SRC_CACHE_DIR)
    if not pesq_src.exists():
        raise RuntimeError("Downloaded PESQ source archive did not contain the expected files.")
    return root


def _build_backend() -> Path:
    src_root = _extract_source()
    src_dir = src_root / "pesq"

    sources = [
        str(_WRAPPER_C),
        str(src_dir / "dsp.c"),
        str(src_dir / "pesqdsp.c"),
        str(src_dir / "pesqmod.c"),
    ]

    include_args = ["-I", str(src_dir)]
    common_flags = ["-shared", "-O2", "-D_CRT_SECURE_NO_WARNINGS"]
    if os.name != "nt":
        common_flags.append("-fPIC")

    build_errors: list[str] = []
    for compiler in _candidate_compilers():
        cmd = compiler + common_flags + ["-o", str(_BACKEND_PATH)] + sources + include_args + ["-lm"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and _BACKEND_PATH.exists():
            return _BACKEND_PATH
        build_errors.append(
            f"$ {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}".strip()
        )

    details = "\n\n".join(build_errors) if build_errors else "No suitable C compiler was found."
    raise RuntimeError(
        "Failed to build the local PESQ backend. Install Zig or set PESQ_CC to a working C compiler.\n"
        f"{details}"
    )


def _load_backend() -> ctypes._CFuncPtr:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    with _LOAD_LOCK:
        if _BACKEND is not None:
            return _BACKEND
        if not _BACKEND_PATH.exists():
            _build_backend()

        lib = ctypes.CDLL(str(_BACKEND_PATH))
        func = lib.pesq_backend
        func.argtypes = [
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        func.restype = ctypes.c_int
        _BACKEND = func
        return _BACKEND


def _check_fs_mode(mode: str, fs: int) -> None:
    if mode not in {"wb", "nb"}:
        raise ValueError("mode should be either 'nb' or 'wb'")
    if fs not in (8000, 16000):
        raise ValueError("fs (sampling frequency) should be either 8000 or 16000")
    if fs == 8000 and mode == "wb":
        raise ValueError("no wide band mode if fs = 8000")


def _as_signal(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("PESQ expects 1D mono signals.")
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _error_from_code(code: int) -> PesqError:
    message = _ERROR_MESSAGES.get(code, _ERROR_MESSAGES[PesqError.UNKNOWN])
    if code == PesqError.INVALID_SAMPLE_RATE:
        return InvalidSampleRateError(message)
    if code in (
        PesqError.OUT_OF_MEMORY_REF,
        PesqError.OUT_OF_MEMORY_DEG,
        PesqError.OUT_OF_MEMORY_TMP,
    ):
        return OutOfMemoryError(message)
    if code == PesqError.BUFFER_TOO_SHORT:
        return BufferTooShortError(message)
    if code == PesqError.NO_UTTERANCES_DETECTED:
        return NoUtterancesError(message)
    return PesqError(message)


def _normalize(ref: np.ndarray, deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_val = float(max(np.max(np.abs(ref)), np.max(np.abs(deg))))
    if max_val <= 0.0:
        max_val = 1.0
    return ref / max_val, deg / max_val


def pesq(fs: int, ref, deg, mode: str = "wb", on_error: int = PesqError.RAISE_EXCEPTION):
    _check_fs_mode(mode, fs)

    ref_arr = _as_signal(ref)
    deg_arr = _as_signal(deg)
    ref_arr, deg_arr = _normalize(ref_arr, deg_arr)

    out_score = ctypes.c_float()
    code = _load_backend()(
        int(fs),
        ref_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(ref_arr.shape[0]),
        deg_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(deg_arr.shape[0]),
        1 if mode == "wb" else 0,
        ctypes.byref(out_score),
    )
    if code == PesqError.SUCCESS:
        return float(out_score.value)
    if on_error == PesqError.RETURN_VALUES:
        return float(code)
    raise _error_from_code(code)


def pesq_batch(
    fs: int,
    ref,
    deg,
    mode: str = "wb",
    n_processor: int | None = None,
    on_error: int = PesqError.RAISE_EXCEPTION,
):
    _check_fs_mode(mode, fs)

    ref_arr = np.asarray(ref, dtype=np.float32)
    deg_arr = np.asarray(deg, dtype=np.float32)

    if ref_arr.ndim == 1:
        if deg_arr.ndim == 1:
            if ref_arr.shape != deg_arr.shape:
                raise ValueError("The shapes of `ref` and `deg` must match.")
            return [pesq(fs, ref_arr, deg_arr, mode=mode, on_error=on_error)]
        if deg_arr.ndim == 2 and ref_arr.shape[-1] == deg_arr.shape[-1]:
            return [pesq(fs, ref_arr, deg_arr[i], mode=mode, on_error=on_error) for i in range(deg_arr.shape[0])]
        raise ValueError("The shape of `deg` is invalid!")

    if ref_arr.ndim == 2:
        if deg_arr.shape != ref_arr.shape:
            raise ValueError("The shape of `deg` is invalid!")
        return [pesq(fs, ref_arr[i], deg_arr[i], mode=mode, on_error=on_error) for i in range(ref_arr.shape[0])]

    raise ValueError("The shape of `ref` should be either 1D or 2D!")
