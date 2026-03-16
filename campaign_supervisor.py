from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from attenuate.locks import ProcessLock


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_now()}] {message}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervise the PESQ campaign until completion.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python-exe", default=None)
    parser.add_argument("--config", default="experiments/pesq_campaign.toml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--restart-delay-seconds", type=int, default=30)
    parser.add_argument("--supervisor-log", default="runs/pesq_campaign.supervisor.log")
    parser.add_argument("--stdout-log", default="runs/pesq_campaign.launch.stdout.log")
    parser.add_argument("--stderr-log", default="runs/pesq_campaign.launch.stderr.log")
    parser.add_argument("--child-pid-file", default="runs/pesq_campaign.pid")
    parser.add_argument("--summary-file", default="runs/pesq_campaign/campaign_summary.json")
    return parser.parse_args()


def _query_gpu_python_processes() -> list[dict[str, str]]:
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
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        process_name = parts[1].lower()
        if "python" not in process_name:
            continue
        rows.append(
            {
                "pid": parts[0],
                "process_name": parts[1],
                "used_gpu_memory": parts[2] if len(parts) > 2 else "",
            }
        )
    return rows


def _wait_for_gpu(supervisor_log: Path, poll_seconds: int) -> None:
    while True:
        gpu_python = _query_gpu_python_processes()
        gpu_python = [row for row in gpu_python if row["pid"] != str(os.getpid())]
        if not gpu_python:
            _append_line(supervisor_log, "gpu_ready")
            return
        _append_line(supervisor_log, f"waiting_for_gpu processes={json.dumps(gpu_python)}")
        time.sleep(max(5, poll_seconds))


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    python_exe = Path(args.python_exe or sys.executable).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    supervisor_log = (repo_root / args.supervisor_log).resolve()
    stdout_log = (repo_root / args.stdout_log).resolve()
    stderr_log = (repo_root / args.stderr_log).resolve()
    child_pid_file = (repo_root / args.child_pid_file).resolve()
    summary_file = (repo_root / args.summary_file).resolve()
    lock = ProcessLock(repo_root / "runs" / "pesq_campaign.supervisor.lock")
    if not lock.acquire():
        _append_line(supervisor_log, "supervisor_duplicate_exit")
        return

    try:
        _append_line(supervisor_log, f"supervisor_start python={python_exe}")

        while not summary_file.exists():
            _wait_for_gpu(supervisor_log, int(args.poll_seconds))

            with stdout_log.open("ab") as stdout_handle, stderr_log.open("ab") as stderr_handle:
                child = subprocess.Popen(
                    [str(python_exe), "run_campaign.py", "--config", str(config_path), "--device", args.device],
                    cwd=repo_root,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
                child_pid_file.write_text(str(child.pid), encoding="utf-8")
                _append_line(supervisor_log, f"campaign_started pid={child.pid}")
                exit_code = child.wait()
            _append_line(supervisor_log, f"campaign_exit pid={child.pid} exit_code={exit_code}")

            if exit_code == 0 and summary_file.exists():
                break
            time.sleep(max(5, int(args.restart_delay_seconds)))

        _append_line(supervisor_log, "supervisor_finished")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
