from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_now()}] {message}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue the PESQ campaign until the GPU is free.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python-exe", default=None)
    parser.add_argument("--config", default="experiments/pesq_campaign.toml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--restart-delay-seconds", type=int, default=30)
    parser.add_argument("--queue-log", default="runs/pesq_campaign.queue.log")
    parser.add_argument("--stdout-log", default="runs/pesq_campaign.launch.stdout.log")
    parser.add_argument("--stderr-log", default="runs/pesq_campaign.launch.stderr.log")
    parser.add_argument("--pid-file", default="runs/pesq_campaign.pid")
    parser.add_argument("--summary-file", default="runs/pesq_campaign/campaign_summary.json")
    return parser.parse_args()


def _gpu_python_processes() -> list[dict[str, str]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    rows: list[dict[str, str]] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 2:
            continue
        pid, process_name = parts[0], parts[1]
        used_memory = parts[2] if len(parts) > 2 else ""
        if not pid.isdigit():
            continue
        normalized_name = process_name.lower()
        if "python" not in normalized_name:
            continue
        rows.append({"pid": pid, "process_name": process_name, "used_gpu_memory": used_memory})
    return rows


def _launch_campaign(
    *,
    repo_root: Path,
    python_exe: str,
    config_path: str,
    device: str,
    stdout_log: Path,
    stderr_log: Path,
    pid_file: Path,
    queue_log: Path,
) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("ab") as stdout_handle, stderr_log.open("ab") as stderr_handle:
        child = subprocess.Popen(
            [python_exe, "run_campaign.py", "--config", config_path, "--device", device],
            cwd=repo_root,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        pid_file.write_text(str(child.pid), encoding="utf-8")
        _append_log(queue_log, f"campaign_started pid={child.pid}")
        return child.wait()


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    python_exe = args.python_exe or sys.executable
    config_path = str((repo_root / args.config).resolve())
    queue_log = (repo_root / args.queue_log).resolve()
    stdout_log = (repo_root / args.stdout_log).resolve()
    stderr_log = (repo_root / args.stderr_log).resolve()
    pid_file = (repo_root / args.pid_file).resolve()
    summary_file = (repo_root / args.summary_file).resolve()

    _append_log(queue_log, "queue_started")

    while not summary_file.exists():
        gpu_python = _gpu_python_processes()
        gpu_python = [row for row in gpu_python if row["pid"] != str(os.getpid())]
        if gpu_python:
            _append_log(queue_log, f"waiting_for_gpu python_gpu_processes={json.dumps(gpu_python)}")
            time.sleep(max(5, int(args.poll_seconds)))
            continue

        exit_code = _launch_campaign(
            repo_root=repo_root,
            python_exe=python_exe,
            config_path=config_path,
            device=args.device,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            pid_file=pid_file,
            queue_log=queue_log,
        )
        _append_log(queue_log, f"campaign_exit exit_code={exit_code}")
        if exit_code == 0 and summary_file.exists():
            break
        time.sleep(max(5, int(args.restart_delay_seconds)))

    _append_log(queue_log, "queue_finished")


if __name__ == "__main__":
    main()
