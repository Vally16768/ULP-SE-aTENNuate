from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from download_voicebank_demand import ensure_voicebank_raw_dataset
from model_to_mcu_specs import RUN_SPECS, find_spec, specs_for_sample_rate
from prepare_voicebank_16k import prepare_voicebank_dataset


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the measured model-to-MCU campaign end-to-end.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-root", default="dataset/voicebank-demand/raw")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wait-for-idle-gpu", action="store_true")
    parser.add_argument("--gpu-max-util", type=int, default=10)
    parser.add_argument("--gpu-max-mem-mib", type=int, default=1024)
    parser.add_argument(
        "--strategy",
        default="16k_first",
        choices=["all", "16k_first"],
        help="Campaign ordering strategy.",
    )
    return parser.parse_args()


def _run(command: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(command, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _read_gpu_state() -> tuple[int, int] | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    line = proc.stdout.strip().splitlines()[0]
    util_s, mem_s = [part.strip() for part in line.split(",")]
    return int(util_s), int(mem_s)


def _wait_for_idle_gpu(max_util: int, max_mem_mib: int) -> None:
    while True:
        state = _read_gpu_state()
        if state is None:
            return
        util, mem = state
        if util <= max_util and mem <= max_mem_mib:
            return
        print(f"[campaign] waiting for idle GPU: util={util}% mem={mem} MiB")
        time.sleep(30)


def _ensure_datasets(raw_root: Path) -> None:
    if not raw_root.exists():
        ensure_voicebank_raw_dataset(raw_root)
    prepare_voicebank_dataset(
        source_root=raw_root,
        out_root=ROOT / "dataset" / "voicebank-demand" / "16k",
        sample_rate=16000,
        overwrite=False,
    )
    prepare_voicebank_dataset(
        source_root=raw_root,
        out_root=ROOT / "dataset" / "voicebank-demand" / "8k",
        sample_rate=8000,
        overwrite=False,
    )


def _evaluate_and_profile(args: argparse.Namespace, spec: dict[str, object]) -> None:
    run_dir = ROOT / str(spec["run_dir"])
    checkpoint = run_dir / "best.pt"
    test_manifest = ROOT / str(spec["test_manifest"])
    test_eval_json = run_dir / "test_eval.json"
    profile_json = run_dir / "profile_raw.json"
    if checkpoint.exists() and (args.force or not test_eval_json.exists()):
        _run(
            [
                args.python,
                "tools/evaluate_checkpoint_manifest.py",
                "--checkpoint",
                checkpoint.as_posix(),
                "--manifest",
                test_manifest.as_posix(),
                "--out-json",
                test_eval_json.as_posix(),
                "--device",
                args.device,
                "--sample-rate",
                str(spec["sample_rate"]),
                "--model-kind",
                str(spec["family"]),
            ],
            cwd=ROOT,
        )
    if checkpoint.exists() and (args.force or not profile_json.exists()):
        _run(
            [
                args.python,
                "tools/profile_checkpoint.py",
                "--checkpoint",
                checkpoint.as_posix(),
                "--out-json",
                profile_json.as_posix(),
                "--device",
                "cpu",
                "--sample-rate",
                str(spec["sample_rate"]),
                "--model-kind",
                str(spec["family"]),
            ],
            cwd=ROOT,
        )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selection_key(payload: dict[str, object]) -> tuple[float, float, float]:
    delta = payload["delta"]
    aggregate = payload["aggregate"]
    return (
        float(delta["delta_PESQ"]),
        float(aggregate["STOI"]),
        float(aggregate["SI_SDR"]),
    )


def _train_spec(args: argparse.Namespace, spec: dict[str, object]) -> None:
    run_dir = ROOT / str(spec["run_dir"])
    summary_path = run_dir / "summary.json"
    if args.skip_train or (summary_path.exists() and not args.force):
        return
    if args.wait_for_idle_gpu and args.device.startswith("cuda"):
        _wait_for_idle_gpu(args.gpu_max_util, args.gpu_max_mem_mib)
    command = [
        args.python,
        "train.py",
        "--config",
        str(spec["config"]),
        "--device",
        args.device,
    ]
    last_state = run_dir / "last_train_state.pt"
    if last_state.exists():
        command.extend(["--resume", last_state.as_posix()])
    _run(command, cwd=ROOT)


def _run_specs(args: argparse.Namespace, specs: list[dict[str, object]]) -> None:
    selected_specs = specs[: args.max_runs] if args.max_runs is not None else specs
    for spec in selected_specs:
        _train_spec(args, spec)
        _evaluate_and_profile(args, spec)


def _choose_best_16k(specs: list[dict[str, object]]) -> dict[str, object]:
    completed: list[tuple[dict[str, object], dict[str, object]]] = []
    for spec in specs:
        run_dir = ROOT / str(spec["run_dir"])
        test_eval_path = run_dir / "test_eval.json"
        if not test_eval_path.exists():
            continue
        completed.append((spec, _read_json(test_eval_path)))
    if not completed:
        raise RuntimeError("No completed 16 kHz runs with test_eval.json available for winner selection.")
    best_spec, _best_payload = max(completed, key=lambda item: _selection_key(item[1]))
    return best_spec


def main() -> None:
    args = parse_args()
    raw_root = ROOT / args.raw_root
    _ensure_datasets(raw_root)
    if args.prepare_only:
        return

    if args.strategy == "all":
        _run_specs(args, RUN_SPECS)
    else:
        specs_16k = specs_for_sample_rate(16000)
        _run_specs(args, specs_16k)
        _run([args.python, "tools/aggregate_model_to_mcu_results.py"], cwd=ROOT)
        best_16k = _choose_best_16k(specs_16k)
        best_family = str(best_16k["family"])
        best_8k = find_spec(family=best_family, sample_rate=8000)
        _run_specs(args, [best_8k])

    _run([args.python, "tools/aggregate_model_to_mcu_results.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
