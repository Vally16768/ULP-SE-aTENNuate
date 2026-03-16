from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_to_mcu_specs import HARDWARE_COST_TIER, RUN_SPECS

REPORTS_DIR = ROOT / "reports"
MEASURED_DIR = REPORTS_DIR / "measured_model_profiles"
TRAINING_JSON = REPORTS_DIR / "training_results_summary.json"
TRAINING_CSV = REPORTS_DIR / "training_results_summary.csv"
TRAINING_MD = REPORTS_DIR / "training_results_summary.md"
MCU_JSON = REPORTS_DIR / "mcu_tradeoff_summary.json"
MCU_CSV = REPORTS_DIR / "mcu_tradeoff_summary.csv"
MCU_MD = REPORTS_DIR / "mcu_tradeoff_summary.md"
FINAL_MD = REPORTS_DIR / "final_embedded_recommendation.md"


QUALITY_CLASS = {
    "atennuate": "offline_waveform",
    "mp_senet_lite": "offline_tf",
    "mp_senet_micro": "causal_waveform_micro",
    "percepnet_class": "causal_hybrid",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def _slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace(".", "").replace("+", "plus")


def _quality_tier(delta_pesq: float) -> str:
    if delta_pesq >= 0.75:
        return "high"
    if delta_pesq >= 0.45:
        return "acceptable"
    if delta_pesq >= 0.20:
        return "degraded"
    return "low"


def _quality_penalty_8k(delta_8k: float | None, delta_16k: float | None) -> str:
    if delta_8k is None or delta_16k is None:
        return "unknown"
    gap = float(delta_16k - delta_8k)
    if gap <= 0.05:
        return "small"
    if gap <= 0.20:
        return "moderate"
    return "large"


def _sort_audio_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row.get("delta_PESQ", float("-inf"))),
        float(row.get("PESQ", float("-inf"))),
        float(row.get("STOI", float("-inf"))),
        float(row.get("SI_SDR", float("-inf"))),
    )


def _sort_candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row.get("delta_PESQ", float("-inf"))),
        float(row.get("STOI", float("-inf"))),
        float(row.get("SI_SDR", float("-inf"))),
        -float(row.get("estimated_energy_per_second_mj", float("inf"))),
        1 if row.get("cost_tier") == "low" else 0,
    )


def _load_completed_runs() -> tuple[list[dict[str, Any]], list[str]]:
    completed: list[dict[str, Any]] = []
    pending: list[str] = []
    for spec in RUN_SPECS:
        run_dir = ROOT / spec["run_dir"]
        summary_path = run_dir / "summary.json"
        best_path = run_dir / "best.pt"
        test_eval_path = run_dir / "test_eval.json"
        profile_path = run_dir / "profile_raw.json"
        if not summary_path.exists() or not best_path.exists() or not test_eval_path.exists() or not profile_path.exists():
            pending.append(spec["name"])
            continue
        summary = _read_json(summary_path)
        test_eval = _read_json(test_eval_path)
        profile = _read_json(profile_path)
        aggregate = dict(test_eval["aggregate"])
        delta = dict(test_eval["delta"])
        completed.append(
            {
                "name": spec["name"],
                "family": spec["family"],
                "sample_rate": spec["sample_rate"],
                "config": spec["config"],
                "run_dir": run_dir.as_posix(),
                "checkpoint": best_path.as_posix(),
                "best_epoch": int(summary["best_epoch"]),
                "PESQ": float(aggregate["PESQ"]),
                "STOI": float(aggregate["STOI"]),
                "SI_SDR": float(aggregate["SI_SDR"]),
                "DELTA_SNR": float(aggregate["DELTA_SNR"]),
                "delta_PESQ": float(delta["delta_PESQ"]),
                "delta_STOI": float(delta["delta_STOI"]),
                "delta_SI_SDR": float(delta["delta_SI_SDR"]),
                "delta_DELTA_SNR": float(delta["delta_DELTA_SNR"]),
                "noisy_baseline_PESQ": float(test_eval["noisy_baseline"]["PESQ"]),
                "profile_raw_path": profile_path.as_posix(),
                "test_eval_path": test_eval_path.as_posix(),
                "profile_raw": profile,
            }
        )
    return completed, pending


def _write_training_reports(rows: list[dict[str, Any]], pending: list[str]) -> None:
    payload = {
        "rows": rows,
        "pending_runs": pending,
    }
    TRAINING_JSON.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with TRAINING_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "family",
                "sample_rate",
                "best_epoch",
                "checkpoint",
                "PESQ",
                "delta_PESQ",
                "STOI",
                "SI_SDR",
                "DELTA_SNR",
                "noisy_baseline_PESQ",
                "profile_raw_path",
                "test_eval_path",
            ]
        )
        for row in sorted(rows, key=lambda item: (item["family"], item["sample_rate"])):
            writer.writerow(
                [
                    row["name"],
                    row["family"],
                    row["sample_rate"],
                    row["best_epoch"],
                    row["checkpoint"],
                    _fmt_float(row["PESQ"]),
                    _fmt_float(row["delta_PESQ"]),
                    _fmt_float(row["STOI"]),
                    _fmt_float(row["SI_SDR"]),
                    _fmt_float(row["DELTA_SNR"]),
                    _fmt_float(row["noisy_baseline_PESQ"]),
                    row["profile_raw_path"],
                    row["test_eval_path"],
                ]
            )

    best_16k = max((row for row in rows if row["sample_rate"] == 16000), key=_sort_audio_key, default=None)
    best_8k = max((row for row in rows if row["sample_rate"] == 8000), key=_sort_audio_key, default=None)
    global_best = max([row for row in (best_16k, best_8k) if row is not None], key=_sort_audio_key, default=None)
    lines = [
        "# Training Results Summary",
        "",
        f"- Completed runs: `{len(rows)}`",
        f"- Pending runs: `{len(pending)}`",
    ]
    if best_16k is not None:
        lines.append(f"- Best 16 kHz: `{best_16k['name']}` with `delta_PESQ {best_16k['delta_PESQ']:.4f}` and `PESQ {best_16k['PESQ']:.4f}`.")
    if best_8k is not None:
        lines.append(f"- Best 8 kHz: `{best_8k['name']}` with `delta_PESQ {best_8k['delta_PESQ']:.4f}` and `PESQ {best_8k['PESQ']:.4f}`.")
    if global_best is not None:
        lines.append(f"- Global SR-normalized leader: `{global_best['name']}`.")
    lines.extend(["", "| Model | SR | PESQ | delta_PESQ | STOI | SI-SDR |", "| --- | --- | --- | --- | --- | --- |"])
    for row in sorted(rows, key=lambda item: (item["family"], item["sample_rate"])):
        lines.append(
            f"| {row['name']} | {row['sample_rate']//1000} kHz | {row['PESQ']:.4f} | {row['delta_PESQ']:.4f} | {row['STOI']:.4f} | {row['SI_SDR']:.4f} |"
        )
    TRAINING_MD.write_text("\n".join(lines), encoding="utf-8")


def _build_measured_profiles(rows: list[dict[str, Any]]) -> None:
    MEASURED_DIR.mkdir(parents=True, exist_ok=True)
    delta_by_family_rate = {(row["family"], row["sample_rate"]): row["delta_PESQ"] for row in rows}
    for row in rows:
        raw = dict(row["profile_raw"])
        family = row["family"]
        sr = int(row["sample_rate"])
        if sr == 8000:
            penalty = _quality_penalty_8k(delta_by_family_rate.get((family, 8000)), delta_by_family_rate.get((family, 16000)))
        else:
            penalty = "none"
        profile = {
            "name": row["name"],
            "family": family,
            "sample_rate": sr,
            "frame_len": int(raw["frame_len"]),
            "hop_len": int(raw["hop_len"]),
            "quality_class": QUALITY_CLASS.get(family, "trained_model"),
            "quality_tier": _quality_tier(float(row["delta_PESQ"])),
            "quality_penalty_estimate": penalty,
            "streaming_mode": raw["streaming_mode"],
            "causal_ready": bool(raw["causal_ready"]),
            "supports_block_inference": bool(raw["supports_block_inference"]),
            "weight_bytes_fp32": int(raw["weight_bytes_fp32"]),
            "weight_bytes_int8": int(raw["weight_bytes_int8"]),
            "runtime_code_bytes": int(raw["runtime_code_bytes"]),
            "activation_peak_bytes": int(raw["activation_peak_bytes"]),
            "workspace_bytes": int(raw["workspace_bytes"]),
            "io_bytes": int(raw["io_bytes"]),
            "op_buckets_per_second": raw["op_buckets_per_second"],
            "accelerator_friendly_buckets": list(raw["accelerator_friendly_buckets"]),
            "notes": f"{raw['notes']} Test delta_PESQ={row['delta_PESQ']:.4f}.",
            "sources": [row["checkpoint"], row["test_eval_path"], row["profile_raw_path"]],
        }
        (MEASURED_DIR / f"{row['name']}.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")


def _run_mcu_simulator() -> dict[str, Any]:
    command = [
        sys.executable,
        "tools/mcu_feasibility_sim.py",
        "--source",
        "measured",
        "--measured-model-dir",
        MEASURED_DIR.as_posix(),
        "--models",
        "all",
        "--hardware",
        "all",
        "--compare-bandwidths",
        "--mode",
        "strict",
        "--out-json",
        MCU_JSON.as_posix(),
        "--out-csv",
        MCU_CSV.as_posix(),
        "--out-md",
        MCU_MD.as_posix(),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return _read_json(MCU_JSON)


def _write_tradeoff_reports(training_rows: list[dict[str, Any]], mcu_payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    training_by_name = {row["name"]: row for row in training_rows}
    hardware_names = sorted({row["hardware"] for row in mcu_payload["rows"]})
    merged_rows: list[dict[str, Any]] = []
    for train_row in sorted(training_rows, key=lambda item: (item["family"], item["sample_rate"])):
        merged = {
            "name": train_row["name"],
            "family": train_row["family"],
            "sample_rate": train_row["sample_rate"],
            "checkpoint": train_row["checkpoint"],
            "PESQ": train_row["PESQ"],
            "delta_PESQ": train_row["delta_PESQ"],
            "STOI": train_row["STOI"],
            "SI_SDR": train_row["SI_SDR"],
            "params": int(train_row["profile_raw"]["num_params"]),
            "int8_flash_bytes": int(train_row["profile_raw"]["weight_bytes_int8"] + train_row["profile_raw"]["runtime_code_bytes"]),
            "peak_sram_bytes": int(train_row["profile_raw"]["activation_peak_bytes"] + train_row["profile_raw"]["workspace_bytes"] + train_row["profile_raw"]["io_bytes"]),
            "latency_ms": float(train_row["profile_raw"]["algorithmic_latency_ms"]),
            "MAC_per_second": float(train_row["profile_raw"]["mac_per_second"]),
        }
        rows = [row for row in mcu_payload["rows"] if row["model"] == train_row["name"]]
        for mcu_row in rows:
            tier = HARDWARE_COST_TIER.get(mcu_row["hardware"], "unknown")
            merged[f"{_slug(mcu_row['hardware'])}_required_mhz"] = mcu_row["required_mhz"]
            merged[f"{_slug(mcu_row['hardware'])}_power_mw"] = mcu_row["estimated_energy_per_second_mj"]
            merged[f"{_slug(mcu_row['hardware'])}_verdict"] = mcu_row["verdict"]
            merged[f"{_slug(mcu_row['hardware'])}_memory_mode"] = mcu_row["memory_mode"]
            merged[f"{_slug(mcu_row['hardware'])}_cost_tier"] = tier
        merged_rows.append(merged)

    with MCU_CSV.open("w", newline="", encoding="utf-8") as handle:
        base_headers = [
            "name",
            "family",
            "sample_rate",
            "checkpoint",
            "PESQ",
            "delta_PESQ",
            "STOI",
            "SI_SDR",
            "params",
            "int8_flash_bytes",
            "peak_sram_bytes",
            "latency_ms",
            "MAC_per_second",
        ]
        dynamic_headers: list[str] = []
        for hardware in hardware_names:
            slug = _slug(hardware)
            dynamic_headers.extend(
                [
                    f"{slug}_required_mhz",
                    f"{slug}_power_mw",
                    f"{slug}_verdict",
                    f"{slug}_memory_mode",
                    f"{slug}_cost_tier",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=base_headers + dynamic_headers)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    best_16k = max((row for row in training_rows if row["sample_rate"] == 16000), key=_sort_audio_key, default=None)
    best_8k = max((row for row in training_rows if row["sample_rate"] == 8000), key=_sort_audio_key, default=None)
    global_best = max([row for row in (best_16k, best_8k) if row is not None], key=_sort_audio_key, default=None)

    pass_rows = [
        {**row, **training_by_name[row["model"]], "cost_tier": HARDWARE_COST_TIER.get(row["hardware"], "unknown")}
        for row in mcu_payload["rows"]
        if row["verdict"] == "PASS" and row["memory_mode"] == "onchip"
    ]
    under_50 = [row for row in pass_rows if float(row["estimated_energy_per_second_mj"]) < 50.0]
    quality_first = max(under_50, key=_sort_candidate_key, default=None)
    if quality_first is None:
        quality_first = max(pass_rows, key=_sort_candidate_key, default=None)
    low_power_first = min(
        pass_rows,
        key=lambda row: (
            float(row["estimated_energy_per_second_mj"]),
            -float(row["delta_PESQ"]),
            -float(row["STOI"]),
            -float(row["SI_SDR"]),
        ),
        default=None,
    )

    mcu_report = {
        "training_rows": training_rows,
        "mcu_rows": mcu_payload["rows"],
        "best_16k": best_16k,
        "best_8k": best_8k,
        "global_best": global_best,
        "best_under_50mw": quality_first,
        "quality_first": quality_first,
        "low_power_first": low_power_first,
        "power_target_met": bool(quality_first is not None and float(quality_first["estimated_energy_per_second_mj"]) < 50.0),
    }
    MCU_JSON.write_text(json.dumps(mcu_report, indent=2), encoding="utf-8")

    lines = [
        "# MCU Tradeoff Summary",
        "",
        f"- Completed measured models: `{len(training_rows)}`",
        f"- Power target met: `{mcu_report['power_target_met']}`",
    ]
    if global_best is not None:
        lines.append(f"- Global SR-normalized leader: `{global_best['name']}` (`delta_PESQ {global_best['delta_PESQ']:.4f}`).")
    if quality_first is not None:
        lines.append(
            f"- Best on-chip candidate under deployment constraints: `{quality_first['model']}` on `{quality_first['hardware']}` at `{quality_first['sample_rate']//1000} kHz`, `required_mhz {quality_first['required_mhz']:.2f}`, `power {quality_first['estimated_energy_per_second_mj']:.2f} mW`."
        )
    lines.extend(["", "| Model | SR | delta_PESQ | PESQ | Params | Flash int8 | SRAM peak | MAC/s |", "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for row in merged_rows:
        lines.append(
            f"| {row['name']} | {row['sample_rate']//1000} kHz | {row['delta_PESQ']:.4f} | {row['PESQ']:.4f} | {row['params']} | {_mib(row['int8_flash_bytes']):.3f} MiB | {_mib(row['peak_sram_bytes']):.3f} MiB | {row['MAC_per_second']:.2f} |"
        )
    MCU_MD.write_text("\n".join(lines), encoding="utf-8")
    return quality_first, low_power_first


def _write_final_recommendation(training_rows: list[dict[str, Any]], quality_first: dict[str, Any] | None, low_power_first: dict[str, Any] | None) -> None:
    best_16k = max((row for row in training_rows if row["sample_rate"] == 16000), key=_sort_audio_key, default=None)
    best_8k = max((row for row in training_rows if row["sample_rate"] == 8000), key=_sort_audio_key, default=None)
    global_best = max([row for row in (best_16k, best_8k) if row is not None], key=_sort_audio_key, default=None)

    lines = [
        "# Final Embedded Recommendation",
        "",
        "## Audio winners",
        "",
    ]
    if best_16k is not None:
        lines.append(f"- Best 16 kHz: `{best_16k['name']}` with `delta_PESQ {best_16k['delta_PESQ']:.4f}`.")
    if best_8k is not None:
        lines.append(f"- Best 8 kHz: `{best_8k['name']}` with `delta_PESQ {best_8k['delta_PESQ']:.4f}`.")
    if global_best is not None:
        lines.append(f"- Best global model independent of sample rate: `{global_best['name']}`.")
    lines.extend(["", "## Deployment recommendations", ""])
    if quality_first is not None:
        lines.append(
            f"- Quality-first: `{quality_first['model']}` on `{quality_first['hardware']}` at `{quality_first['sample_rate']//1000} kHz`, `power {quality_first['estimated_energy_per_second_mj']:.2f} mW`, `required_mhz {quality_first['required_mhz']:.2f}`, `cost tier {quality_first['cost_tier']}`."
        )
    if low_power_first is not None:
        lines.append(
            f"- Low-power-first: `{low_power_first['model']}` on `{low_power_first['hardware']}` at `{low_power_first['sample_rate']//1000} kHz`, `power {low_power_first['estimated_energy_per_second_mj']:.2f} mW`, `required_mhz {low_power_first['required_mhz']:.2f}`, `cost tier {low_power_first['cost_tier']}`."
        )
    FINAL_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    completed_rows, pending = _load_completed_runs()
    _write_training_reports(completed_rows, pending)
    if not completed_rows:
        print("No completed runs with test_eval.json + profile_raw.json available yet.")
        return
    _build_measured_profiles(completed_rows)
    mcu_payload = _run_mcu_simulator()
    quality_first, low_power_first = _write_tradeoff_reports(completed_rows, mcu_payload)
    _write_final_recommendation(completed_rows, quality_first, low_power_first)
    print(f"Wrote {TRAINING_JSON}")
    print(f"Wrote {MCU_JSON}")
    print(f"Wrote {FINAL_MD}")


if __name__ == "__main__":
    main()
