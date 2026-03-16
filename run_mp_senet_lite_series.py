from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


RUNS = [
    {
        "name": "mp_senet_lite_voicebank",
        "config": "experiments/mp_senet_lite_voicebank.toml",
        "depends_on": None,
    },
    {
        "name": "mp_senet_lite_voicebank_spectral_gate_ft",
        "config": "experiments/mp_senet_lite_voicebank_spectral_gate_ft.toml",
        "depends_on": "runs/mp_senet_lite_voicebank/best.pt",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full MP-SENet-lite training series.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    parser.add_argument("--device", default="cuda", help="Training device.")
    parser.add_argument("--out-dir", default="runs/mp_senet_lite_series", help="Directory for logs and summary.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "series_summary.json"
    summary: list[dict[str, object]] = []

    for spec in RUNS:
        if spec["depends_on"] is not None and not Path(spec["depends_on"]).exists():
            raise FileNotFoundError(f"Dependency checkpoint missing for {spec['name']}: {spec['depends_on']}")

        stdout_path = out_dir / f"{spec['name']}.stdout.log"
        stderr_path = out_dir / f"{spec['name']}.stderr.log"
        command = [
            args.python,
            "train.py",
            "--config",
            spec["config"],
            "--device",
            args.device,
        ]

        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            proc = subprocess.run(command, stdout=stdout_handle, stderr=stderr_handle, text=True)

        entry = {
            "name": spec["name"],
            "config": spec["config"],
            "returncode": int(proc.returncode),
            "stdout": stdout_path.as_posix(),
            "stderr": stderr_path.as_posix(),
        }
        summary.append(entry)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if proc.returncode != 0:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
