from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS = "PASS"
PARTIAL = "PARTIAL"
FAIL = "FAIL"

QUALITY_SCORE = {"high": 3, "acceptable": 2, "degraded": 1, "low": 0}
VERDICT_SCORE = {PASS: 2, PARTIAL: 1, FAIL: 0}
COMPUTE_BUCKETS = (
    "conv_int8",
    "recurrent",
    "matmul_attention",
    "fft_stft",
    "fft_conv",
    "frontend_dsp",
    "elementwise",
)
ACCELERATOR_CANDIDATE_BUCKETS = {"conv_int8", "recurrent", "matmul_attention"}
LATENCY_PASS_MS = 40.0
LATENCY_PARTIAL_MS = 150.0
FLASH_PASS_RATIO = 0.80
SRAM_PASS_RATIO = 0.70
COMPUTE_PARTIAL_RTF = 1.25


@dataclass(frozen=True)
class ModelProfile:
    name: str
    family: str
    sample_rate: int
    frame_len: int
    hop_len: int
    quality_class: str
    quality_tier: str
    quality_penalty_estimate: str
    streaming_mode: str
    causal_ready: bool
    supports_block_inference: bool
    weight_bytes_fp32: int
    weight_bytes_int8: int
    runtime_code_bytes: int
    activation_peak_bytes: int
    workspace_bytes: int
    io_bytes: int
    op_buckets_per_second: dict[str, float]
    accelerator_friendly_buckets: list[str]
    notes: str
    sources: list[str]


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    family: str
    category: str
    core: str
    freq_mhz: float
    sram_bytes: int
    flash_bytes: int
    cpu_mac_per_cycle: float
    cpu_efficiency: float
    dsp_mac_per_cycle: float
    dsp_efficiency: float
    active_current_uA_per_MHz: float
    supply_voltage_v: float
    npu_gops: float
    npu_efficiency: float
    npu_power_mw_at_full: float
    npu_supported_buckets: list[str]
    supports_external_memory: bool
    on_chip_only: bool
    external_flash_bytes: int
    external_sram_bytes: int
    external_memory_penalty_factor: float
    external_memory_power_mw: float
    notes: str
    sources: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate MCU feasibility for speech enhancement models.")
    parser.add_argument("--models", default="all", help="Comma-separated model profile names or 'all'.")
    parser.add_argument("--hardware", default="all", help="Comma-separated hardware profile names or 'all'.")
    parser.add_argument("--profiles-dir", default="profiles", help="Root folder for profiles.")
    parser.add_argument("--source", default="static", choices=["static", "measured"], help="Read model profiles from static profiles/ or measured post-training JSONs.")
    parser.add_argument("--measured-model-dir", default="reports/measured_model_profiles", help="Directory with measured model profile JSON files.")
    parser.add_argument("--compare-bandwidths", action="store_true", help="Emit 16 kHz vs 8 kHz comparisons.")
    parser.add_argument("--mode", default="strict", choices=["strict", "stretch", "porting-target"], help="Evaluation mode.")
    parser.add_argument("--out-json", default="reports/mcu_feasibility.json")
    parser.add_argument("--out-csv", default="reports/mcu_feasibility.csv")
    parser.add_argument("--out-md", default="reports/mcu_feasibility.md")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_profiles_from_dir(model_dir: Path) -> dict[str, ModelProfile]:
    out: dict[str, ModelProfile] = {}
    for path in sorted(model_dir.glob("*.json")):
        raw = _read_json(path)
        required = {"sample_rate", "frame_len", "hop_len", "op_buckets_per_second"}
        if not required.issubset(raw):
            continue
        family = str(raw.get("family") or raw.get("model_config", {}).get("kind") or path.stem)
        out[path.stem] = ModelProfile(
            name=str(raw["name"]),
            family=family,
            sample_rate=int(raw["sample_rate"]),
            frame_len=int(raw["frame_len"]),
            hop_len=int(raw["hop_len"]),
            quality_class=str(raw.get("quality_class", family)),
            quality_tier=str(raw.get("quality_tier", "unknown")),
            quality_penalty_estimate=str(raw.get("quality_penalty_estimate", "unknown")),
            streaming_mode=str(raw["streaming_mode"]),
            causal_ready=bool(raw["causal_ready"]),
            supports_block_inference=bool(raw["supports_block_inference"]),
            weight_bytes_fp32=int(raw["weight_bytes_fp32"]),
            weight_bytes_int8=int(raw["weight_bytes_int8"]),
            runtime_code_bytes=int(raw["runtime_code_bytes"]),
            activation_peak_bytes=int(raw["activation_peak_bytes"]),
            workspace_bytes=int(raw["workspace_bytes"]),
            io_bytes=int(raw.get("io_bytes", 0)),
            op_buckets_per_second={k: float(v) for k, v in raw["op_buckets_per_second"].items()},
            accelerator_friendly_buckets=[str(v) for v in raw.get("accelerator_friendly_buckets", [])],
            notes=str(raw.get("notes", "")),
            sources=[str(v) for v in raw.get("sources", [])],
        )
    return out


def load_model_profiles(root: Path) -> dict[str, ModelProfile]:
    return load_model_profiles_from_dir(root / "models")


def load_hardware_profiles(root: Path) -> dict[str, HardwareProfile]:
    out: dict[str, HardwareProfile] = {}
    for path in sorted((root / "hardware").glob("*.json")):
        raw = _read_json(path)
        out[path.stem] = HardwareProfile(
            name=str(raw["name"]),
            family=str(raw["family"]),
            category=str(raw["category"]),
            core=str(raw["core"]),
            freq_mhz=float(raw["freq_mhz"]),
            sram_bytes=int(raw["sram_bytes"]),
            flash_bytes=int(raw["flash_bytes"]),
            cpu_mac_per_cycle=float(raw["cpu_mac_per_cycle"]),
            cpu_efficiency=float(raw["cpu_efficiency"]),
            dsp_mac_per_cycle=float(raw["dsp_mac_per_cycle"]),
            dsp_efficiency=float(raw["dsp_efficiency"]),
            active_current_uA_per_MHz=float(raw["active_current_uA_per_MHz"]),
            supply_voltage_v=float(raw["supply_voltage_v"]),
            npu_gops=float(raw.get("npu_gops", 0.0)),
            npu_efficiency=float(raw.get("npu_efficiency", 0.0)),
            npu_power_mw_at_full=float(raw.get("npu_power_mw_at_full", 0.0)),
            npu_supported_buckets=[str(v) for v in raw.get("npu_supported_buckets", [])],
            supports_external_memory=bool(raw.get("supports_external_memory", False)),
            on_chip_only=bool(raw.get("on_chip_only", True)),
            external_flash_bytes=int(raw.get("external_flash_bytes", 0)),
            external_sram_bytes=int(raw.get("external_sram_bytes", 0)),
            external_memory_penalty_factor=float(raw.get("external_memory_penalty_factor", 1.0)),
            external_memory_power_mw=float(raw.get("external_memory_power_mw", 0.0)),
            notes=str(raw.get("notes", "")),
            sources=[str(v) for v in raw.get("sources", [])],
        )
    return out


def select_profiles(items: dict[str, Any], selector: str) -> list[Any]:
    if selector.strip().lower() == "all":
        return list(items.values())
    wanted = {token.strip() for token in selector.split(",") if token.strip()}
    selected: list[Any] = []
    missing: list[str] = []
    for key in sorted(wanted):
        if key in items:
            selected.append(items[key])
        else:
            missing.append(key)
    if missing:
        raise SystemExit(f"Unknown profiles: {', '.join(missing)}")
    return selected


def mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def throughput_cpu(hw: HardwareProfile) -> float:
    return hw.freq_mhz * 1e6 * hw.cpu_mac_per_cycle * hw.cpu_efficiency


def throughput_dsp(hw: HardwareProfile) -> float:
    return hw.freq_mhz * 1e6 * hw.dsp_mac_per_cycle * hw.dsp_efficiency


def throughput_npu(hw: HardwareProfile) -> float:
    return hw.npu_gops * 1e9 * hw.npu_efficiency


def pick_engine(bucket: str, hw: HardwareProfile) -> tuple[str, float]:
    cpu_tp = throughput_cpu(hw)
    dsp_tp = throughput_dsp(hw)
    npu_tp = throughput_npu(hw)
    npu_supported = bucket in hw.npu_supported_buckets and npu_tp > 0

    if bucket in {"fft_stft", "fft_conv", "frontend_dsp"}:
        if dsp_tp >= cpu_tp and dsp_tp > 0:
            return "dsp", dsp_tp
        return "cpu", cpu_tp

    if bucket == "matmul_attention":
        if npu_supported:
            return "npu", npu_tp
        return "cpu", cpu_tp

    if bucket in {"conv_int8", "recurrent"}:
        if npu_supported:
            return "npu", npu_tp
        if dsp_tp >= cpu_tp and dsp_tp > 0:
            return "dsp", dsp_tp
        return "cpu", cpu_tp

    return "cpu", cpu_tp


def ratio_state(ratio: float, pass_threshold: float, partial_threshold: float) -> str:
    if ratio <= pass_threshold:
        return PASS
    if ratio <= partial_threshold:
        return PARTIAL
    return FAIL


def latency_state(latency_ms: float) -> str:
    if latency_ms <= LATENCY_PASS_MS:
        return PASS
    if latency_ms <= LATENCY_PARTIAL_MS:
        return PARTIAL
    return FAIL


def verdict_from_states(states: list[str]) -> str:
    if FAIL in states:
        return FAIL
    if PARTIAL in states:
        return PARTIAL
    return PASS


def evaluate_pair(model: ModelProfile, hw: HardwareProfile, mode: str) -> dict[str, Any]:
    flash_bytes = model.weight_bytes_int8 + model.runtime_code_bytes
    sram_bytes = model.activation_peak_bytes + model.workspace_bytes + model.io_bytes
    effective_flash_limit = hw.flash_bytes
    effective_sram_limit = hw.sram_bytes
    uses_external_flash = False
    uses_external_sram = False

    if mode == "stretch" and hw.supports_external_memory:
        if flash_bytes > hw.flash_bytes and hw.external_flash_bytes > 0:
            effective_flash_limit = max(hw.flash_bytes, hw.external_flash_bytes)
            uses_external_flash = flash_bytes <= effective_flash_limit
        if sram_bytes > hw.sram_bytes and hw.external_sram_bytes > 0:
            effective_sram_limit = max(hw.sram_bytes, hw.external_sram_bytes)
            uses_external_sram = sram_bytes <= effective_sram_limit

    flash_ratio = math.inf if effective_flash_limit <= 0 and flash_bytes > 0 else (flash_bytes / effective_flash_limit if effective_flash_limit > 0 else 0.0)
    sram_ratio = math.inf if effective_sram_limit <= 0 and sram_bytes > 0 else (sram_bytes / effective_sram_limit if effective_sram_limit > 0 else 0.0)

    hop_ms = 1000.0 * model.hop_len / model.sample_rate
    algorithmic_latency_ms = 1000.0 * model.frame_len / model.sample_rate

    cpu_time = 0.0
    dsp_time = 0.0
    npu_time = 0.0
    total_ops = 0.0
    unsupported_ops = 0.0
    bucket_rows: list[dict[str, Any]] = []

    for bucket in COMPUTE_BUCKETS:
        ops = float(model.op_buckets_per_second.get(bucket, 0.0))
        if ops <= 0:
            continue
        engine, tp = pick_engine(bucket, hw)
        time_s = math.inf if tp <= 0 else (ops / tp)
        total_ops += ops
        if engine == "cpu":
            cpu_time += time_s
        elif engine == "dsp":
            dsp_time += time_s
        elif engine == "npu":
            npu_time += time_s
        if hw.npu_gops > 0 and bucket in ACCELERATOR_CANDIDATE_BUCKETS and engine != "npu":
            unsupported_ops += ops
        if hw.npu_gops <= 0:
            unsupported_ops += ops
        bucket_rows.append({
            "bucket": bucket,
            "ops_per_second": ops,
            "engine": engine,
            "throughput_ops_per_second": tp,
            "audio_seconds_per_second": time_s,
        })

    rtf = cpu_time + dsp_time + npu_time
    uses_external_memory = uses_external_flash or uses_external_sram
    cpu_dsp_time = cpu_time + dsp_time
    npu_time_effective = npu_time
    if uses_external_memory:
        rtf *= hw.external_memory_penalty_factor
        cpu_dsp_time *= hw.external_memory_penalty_factor
        npu_time_effective *= hw.external_memory_penalty_factor
    compute_time_per_hop_ms = rtf * hop_ms
    unsupported_fraction = (unsupported_ops / total_ops) if total_ops > 0 else 0.0

    if npu_time_effective >= 1.0 and cpu_dsp_time > 0:
        required_mhz = math.inf
    else:
        denom = max(1e-9, 1.0 - npu_time_effective)
        scale = max(1.0, cpu_dsp_time / denom)
        required_mhz = hw.freq_mhz * scale

    base_power_mw = hw.supply_voltage_v * hw.active_current_uA_per_MHz * hw.freq_mhz / 1000.0
    cpu_dsp_load = min(1.0, cpu_time + dsp_time)
    npu_load = min(1.0, npu_time)
    estimated_energy_per_second_mj = (base_power_mw * cpu_dsp_load) + (hw.npu_power_mw_at_full * npu_load)
    if uses_external_memory:
        estimated_energy_per_second_mj += hw.external_memory_power_mw

    flash_fit_state = ratio_state(flash_ratio, FLASH_PASS_RATIO, 1.0)
    sram_fit_state = ratio_state(sram_ratio, SRAM_PASS_RATIO, 1.0)
    compute_fit_state = PASS if rtf <= 1.0 else PARTIAL if rtf <= COMPUTE_PARTIAL_RTF else FAIL
    latency_fit_state = latency_state(algorithmic_latency_ms)
    verdict = verdict_from_states([flash_fit_state, sram_fit_state, compute_fit_state, latency_fit_state])

    reasons: list[str] = []
    for label, state in (
        ("flash", flash_fit_state),
        ("sram", sram_fit_state),
        ("compute", compute_fit_state),
        ("latency", latency_fit_state),
    ):
        if state != PASS:
            reasons.append(label)
    if model.streaming_mode == "offline":
        reasons.append("offline_only")
    if mode == "strict" and hw.flash_bytes <= 0 and flash_bytes > 0:
        reasons.append("flashless_onchip")
    reasons = sorted(set(reasons))

    return {
        "model": model.name,
        "family": model.family,
        "sample_rate": model.sample_rate,
        "quality_class": model.quality_class,
        "quality_tier": model.quality_tier,
        "quality_penalty_estimate": model.quality_penalty_estimate,
        "streaming_mode": model.streaming_mode,
        "causal_ready": model.causal_ready,
        "supports_block_inference": model.supports_block_inference,
        "hardware": hw.name,
        "hardware_family": hw.family,
        "hardware_category": hw.category,
        "core": hw.core,
        "verdict": verdict,
        "reasons": reasons,
        "mode": mode,
        "memory_mode": "external" if uses_external_memory else "onchip",
        "uses_external_flash": uses_external_flash,
        "uses_external_sram": uses_external_sram,
        "flash_bytes": flash_bytes,
        "flash_limit_bytes": effective_flash_limit,
        "flash_ratio": flash_ratio,
        "sram_peak_bytes": sram_bytes,
        "sram_limit_bytes": effective_sram_limit,
        "sram_ratio": sram_ratio,
        "workspace_bytes": model.workspace_bytes,
        "algorithmic_latency_ms": algorithmic_latency_ms,
        "hop_ms": hop_ms,
        "compute_time_per_hop_ms": compute_time_per_hop_ms,
        "real_time_factor": rtf,
        "required_mhz": required_mhz,
        "estimated_energy_per_second_mj": estimated_energy_per_second_mj,
        "unsupported_op_fraction": unsupported_fraction,
        "weight_bytes_fp32": model.weight_bytes_fp32,
        "weight_bytes_int8": model.weight_bytes_int8,
        "runtime_code_bytes": model.runtime_code_bytes,
        "bucket_rows": bucket_rows,
        "total_ops_per_second": total_ops,
        "cpu_audio_seconds_per_second": cpu_time,
        "dsp_audio_seconds_per_second": dsp_time,
        "npu_audio_seconds_per_second": npu_time,
        "model_notes": model.notes,
        "hardware_notes": hw.notes,
    }


def compare_bandwidths(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["family"], row["hardware"]), {})[int(row["sample_rate"])] = row

    comparisons: list[dict[str, Any]] = []
    for (family, hardware), group in sorted(grouped.items()):
        row16 = group.get(16000)
        row8 = group.get(8000)
        if row16 is None or row8 is None:
            continue
        delta_compute = row8["real_time_factor"] / row16["real_time_factor"] if row16["real_time_factor"] > 0 else math.inf
        delta_sram = row8["sram_peak_bytes"] / row16["sram_peak_bytes"] if row16["sram_peak_bytes"] > 0 else math.inf
        delta_flash = row8["flash_bytes"] / row16["flash_bytes"] if row16["flash_bytes"] > 0 else math.inf
        delta_latency = row8["algorithmic_latency_ms"] / row16["algorithmic_latency_ms"] if row16["algorithmic_latency_ms"] > 0 else math.inf
        best_gain = max(1.0 - delta_compute, 1.0 - delta_sram)
        deployment_gain = "large" if best_gain >= 0.50 else "medium" if best_gain >= 0.25 else "small"

        preferred_bandwidth = "balanced"
        if VERDICT_SCORE[row8["verdict"]] > VERDICT_SCORE[row16["verdict"]]:
            preferred_bandwidth = "8 kHz preferred"
        elif VERDICT_SCORE[row16["verdict"]] > VERDICT_SCORE[row8["verdict"]]:
            preferred_bandwidth = "16 kHz preferred"
        else:
            q16 = QUALITY_SCORE.get(row16["quality_tier"], -1)
            q8 = QUALITY_SCORE.get(row8["quality_tier"], -1)
            if q16 > q8:
                preferred_bandwidth = "16 kHz preferred"
            elif q8 > q16:
                preferred_bandwidth = "8 kHz preferred"
            elif row8["estimated_energy_per_second_mj"] < row16["estimated_energy_per_second_mj"]:
                preferred_bandwidth = "8 kHz preferred"
            elif row16["estimated_energy_per_second_mj"] < row8["estimated_energy_per_second_mj"]:
                preferred_bandwidth = "16 kHz preferred"

        comparisons.append({
            "family": family,
            "hardware": hardware,
            "verdict_16k": row16["verdict"],
            "verdict_8k": row8["verdict"],
            "delta_compute": delta_compute,
            "delta_sram": delta_sram,
            "delta_flash": delta_flash,
            "delta_latency": delta_latency,
            "quality_penalty_estimate": row8["quality_penalty_estimate"],
            "quality_tier_16k": row16["quality_tier"],
            "quality_tier_8k": row8["quality_tier"],
            "relative_deployment_gain": deployment_gain,
            "preferred_bandwidth": preferred_bandwidth,
        })
    return comparisons


def is_streaming_candidate(row: dict[str, Any]) -> bool:
    return bool(row["causal_ready"]) and str(row["streaming_mode"]) == "streaming"


def row_sort_key_quality(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        VERDICT_SCORE[row["verdict"]],
        1 if row.get("memory_mode") == "onchip" else 0,
        QUALITY_SCORE.get(row["quality_tier"], -1),
        -row["estimated_energy_per_second_mj"],
        -row["real_time_factor"],
    )


def row_sort_key_efficiency(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        VERDICT_SCORE[row["verdict"]],
        1 if row.get("memory_mode") == "onchip" else 0,
        -row["estimated_energy_per_second_mj"],
        -row["real_time_factor"],
        1 if row["sample_rate"] == 8000 else 0,
        QUALITY_SCORE.get(row["quality_tier"], -1),
    )


def pick_best(rows: list[dict[str, Any]], *, category: str | None, sample_rate: int, key_fn) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["sample_rate"] == sample_rate and (category is None or row["hardware_category"] == category)]
    if not candidates:
        return None
    return sorted(candidates, key=key_fn, reverse=True)[0]


def pick_overall(rows: list[dict[str, Any]], key_fn) -> dict[str, Any] | None:
    return sorted(rows, key=key_fn, reverse=True)[0] if rows else None


def recommendation_block(rows: list[dict[str, Any]], primary: dict[str, Any] | None, key_fn) -> dict[str, Any] | None:
    if primary is None:
        return None
    alternatives = [row for row in rows if row is not primary and row["sample_rate"] != primary["sample_rate"]]
    alt = sorted(alternatives, key=key_fn, reverse=True)[0] if alternatives else None
    return {"primary": primary, "alternative": alt}


def porting_sort_key_quality(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        VERDICT_SCORE[row["verdict"]],
        1 if row.get("memory_mode") == "onchip" else 0,
        QUALITY_SCORE.get(row["quality_tier"], -1),
        -row["estimated_energy_per_second_mj"],
        -row["real_time_factor"],
    )


def porting_sort_key_efficiency(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        VERDICT_SCORE[row["verdict"]],
        1 if row.get("memory_mode") == "onchip" else 0,
        1 if row["sample_rate"] == 8000 else 0,
        -row["estimated_energy_per_second_mj"],
        -row["real_time_factor"],
        QUALITY_SCORE.get(row["quality_tier"], -1),
    )


def best_by_family(rows: list[dict[str, Any]], family: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["family"] == family]
    if not candidates:
        return None
    return sorted(candidates, key=row_sort_key_quality, reverse=True)[0]


def build_porting_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    streaming_rows = [row for row in rows if is_streaming_candidate(row)]
    primary_quality = pick_overall(streaming_rows, porting_sort_key_quality)
    primary_efficiency = pick_overall(streaming_rows, porting_sort_key_efficiency)
    classic_target = pick_best(streaming_rows, category="classic_mcu", sample_rate=16000, key_fn=porting_sort_key_quality)
    if classic_target is None or classic_target["verdict"] != PASS:
        classic_target = pick_best(streaming_rows, category="classic_mcu", sample_rate=8000, key_fn=porting_sort_key_efficiency)
    mcu_npu_target = pick_best(streaming_rows, category="mcu_npu", sample_rate=16000, key_fn=porting_sort_key_quality)
    if mcu_npu_target is None:
        mcu_npu_target = pick_best(streaming_rows, category="mcu_npu", sample_rate=8000, key_fn=porting_sort_key_efficiency)

    repo_constraints: list[dict[str, Any]] = []
    for family in ("atennuate", "mp_senet_lite"):
        best = best_by_family(rows, family)
        if best is None:
            continue
        repo_constraints.append({
            "family": family,
            "best_candidate": best["model"],
            "hardware": best["hardware"],
            "sample_rate": best["sample_rate"],
            "verdict": best["verdict"],
            "memory_mode": best["memory_mode"],
            "reasons": best["reasons"],
            "recommendation": "redesign_to_streaming_causal" if "offline_only" in best["reasons"] else "quantize_or_reduce",
        })

    summary = {
        "quality_first_target": recommendation_block(streaming_rows, primary_quality, porting_sort_key_quality),
        "efficiency_first_target": recommendation_block(streaming_rows, primary_efficiency, porting_sort_key_efficiency),
        "classic_mcu_target": classic_target,
        "mcu_npu_target": mcu_npu_target,
        "repo_model_constraints": repo_constraints,
    }
    return summary


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    if args.mode == "porting-target":
        if str(out_json).replace("\\", "/") == "reports/mcu_feasibility.json":
            out_json = Path("reports/mcu_porting_target.json")
        if str(out_csv).replace("\\", "/") == "reports/mcu_feasibility.csv":
            out_csv = Path("reports/mcu_porting_target.csv")
        if str(out_md).replace("\\", "/") == "reports/mcu_feasibility.md":
            out_md = Path("reports/mcu_porting_target.md")
    return out_json, out_csv, out_md


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "model", "family", "sample_rate", "hardware", "hardware_category", "verdict", "quality_tier",
            "flash_bytes", "flash_limit_bytes", "sram_peak_bytes", "sram_limit_bytes", "algorithmic_latency_ms",
            "compute_time_per_hop_ms", "real_time_factor", "required_mhz", "estimated_energy_per_second_mj",
            "unsupported_op_fraction", "memory_mode", "reasons",
        ])
        for row in rows:
            writer.writerow([
                row["model"], row["family"], row["sample_rate"], row["hardware"], row["hardware_category"],
                row["verdict"], row["quality_tier"], row["flash_bytes"], row["flash_limit_bytes"],
                row["sram_peak_bytes"], row["sram_limit_bytes"], f"{row['algorithmic_latency_ms']:.2f}",
                f"{row['compute_time_per_hop_ms']:.2f}", f"{row['real_time_factor']:.4f}", "inf" if math.isinf(row["required_mhz"]) else f"{row['required_mhz']:.2f}",
                f"{row['estimated_energy_per_second_mj']:.4f}", f"{row['unsupported_op_fraction']:.4f}", row["memory_mode"],
                ",".join(row["reasons"]),
            ])


def fmt_bytes(value: int) -> str:
    return f"{mib(value):.2f} MiB"


def fmt_ratio(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.2f}x"


def fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def build_markdown(
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    recommendations: dict[str, Any],
    mode: str,
    porting_target: dict[str, Any] | None,
) -> str:
    lines: list[str] = [
        "# MCU Low-Power Feasibility Report",
        "",
        f"This report compares 16 kHz and 8 kHz deployment profiles for the same model families and target MCUs. Evaluation mode: `{mode}`.",
        "",
        "## Final Recommendations",
        "",
        "### Quality-first recommendation",
    ]

    def recommendation_text(block: dict[str, Any] | None) -> list[str]:
        if block is None:
            return ["No feasible recommendation."]
        primary = block["primary"]
        alt = block.get("alternative")
        out = [f"Primary: `{primary['model']}` on `{primary['hardware']}` at `{primary['sample_rate']//1000} kHz` -> `{primary['verdict']}`."]
        if alt is not None:
            out.append(f"Alternative: `{alt['model']}` on `{alt['hardware']}` at `{alt['sample_rate']//1000} kHz` -> `{alt['verdict']}`.")
        return out

    lines.extend(recommendation_text(recommendations.get("quality_first")))
    lines.extend(["", "### Efficiency-first recommendation"])
    lines.extend(recommendation_text(recommendations.get("efficiency_first")))
    lines.append("")

    if porting_target is not None:
        lines.extend(["## Porting Target", ""])
        q = porting_target.get("quality_first_target")
        e = porting_target.get("efficiency_first_target")
        classic = porting_target.get("classic_mcu_target")
        npu = porting_target.get("mcu_npu_target")
        lines.append("### Quality-first embedded target")
        lines.extend(recommendation_text(q))
        lines.append("")
        lines.append("### Efficiency-first embedded target")
        lines.extend(recommendation_text(e))
        lines.append("")
        target_rows: list[list[str]] = []
        for label, row in (("classic_mcu_target", classic), ("mcu_npu_target", npu)):
            if row is None:
                target_rows.append([label, "none", "-", "-", "-", "-"])
            else:
                target_rows.append([
                    label,
                    row["model"],
                    row["hardware"],
                    f"{row['sample_rate']//1000} kHz",
                    row["memory_mode"],
                    row["verdict"],
                ])
        lines.extend([
            fmt_table(["Target", "Model", "Hardware", "Bandwidth", "Memory", "Verdict"], target_rows),
            "",
        ])
        repo_rows = []
        for item in porting_target.get("repo_model_constraints", []):
            repo_rows.append([
                item["family"],
                item["best_candidate"],
                item["hardware"],
                f"{item['sample_rate']//1000} kHz",
                item["verdict"],
                ",".join(item["reasons"]),
                item["recommendation"],
            ])
        lines.extend([
            "### Current Repo Model Fit",
            fmt_table(["Family", "Best candidate", "Hardware", "Bandwidth", "Verdict", "Reasons", "Action"], repo_rows),
            "",
        ])

    shortlist_rows: list[list[str]] = []
    for label, item in [
        ("best classic MCU @16 kHz", recommendations.get("best_classic_16k")),
        ("best classic MCU @8 kHz", recommendations.get("best_classic_8k")),
        ("best MCU+NPU @16 kHz", recommendations.get("best_mcu_npu_16k")),
        ("best MCU+NPU @8 kHz", recommendations.get("best_mcu_npu_8k")),
    ]:
        if item is None:
            shortlist_rows.append([label, "none", "-", "-", "-"])
        else:
            shortlist_rows.append([label, item["model"], item["hardware"], f"{item['sample_rate']//1000} kHz", item["verdict"]])
    lines.extend(["## Shortlist", "", fmt_table(["Category", "Model", "Hardware", "Bandwidth", "Verdict"], shortlist_rows), ""])

    comparison_rows = [
        [
            comp["family"], comp["hardware"], comp["verdict_16k"], comp["verdict_8k"], fmt_ratio(comp["delta_compute"]),
            fmt_ratio(comp["delta_sram"]), comp["quality_penalty_estimate"], comp["relative_deployment_gain"], comp["preferred_bandwidth"],
        ]
        for comp in comparisons
    ]
    lines.extend([
        "## 16 kHz vs 8 kHz By Family And Hardware",
        "",
        fmt_table(
            ["Family", "Hardware", "16 kHz", "8 kHz", "Compute 8/16", "SRAM 8/16", "Quality penalty", "Deployment gain", "Preference"],
            comparison_rows,
        ),
        "",
    ])

    for hardware in sorted({row["hardware"] for row in rows}):
        lines.extend([f"## {hardware}", ""])
        for sample_rate in (16000, 8000):
            bucket = sorted([row for row in rows if row["hardware"] == hardware and row["sample_rate"] == sample_rate], key=row_sort_key_quality, reverse=True)
            table_rows = [
                [
                    row["model"], row["verdict"], row["quality_tier"], fmt_bytes(row["flash_bytes"]), fmt_bytes(row["sram_peak_bytes"]),
                    f"{row['algorithmic_latency_ms']:.1f}", f"{row['real_time_factor']:.2f}", row["memory_mode"], ",".join(row["reasons"]) or "-",
                ]
                for row in bucket
            ]
            lines.extend([
                f"### {sample_rate // 1000} kHz candidates",
                fmt_table(["Model", "Verdict", "Quality", "Flash", "SRAM", "Latency ms", "RTF", "Memory", "Reasons"], table_rows),
                "",
            ])

    lines.extend([
        "## Notes",
        "",
        "- Strict mode means on-chip only. Stretch mode allows external flash and/or SRAM when the hardware profile exposes it.",
        "- Porting-target mode evaluates feasibility with stretch rules, then narrows recommendations to streaming and causal candidates.",
        "- 16 kHz and 8 kHz PESQ are not compared as absolute cross-band numbers. The report uses quality tiers and deployment gains instead.",
        "- Estimated power is a coarse simulator output derived from core current, modeled engine load, configured NPU power, and optional external-memory penalty. It is useful for ranking, not for final power sign-off.",
        "- aTENNuate and MP-SENet-lite current repo variants are treated as offline blocks, so algorithmic latency is a hard constraint.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    profiles_root = Path(args.profiles_dir)
    model_dir = Path(args.measured_model_dir) if args.source == "measured" else (profiles_root / "models")
    models = select_profiles(load_model_profiles_from_dir(model_dir), args.models)
    hardware = select_profiles(load_hardware_profiles(profiles_root), args.hardware)

    eval_mode = "stretch" if args.mode == "porting-target" else args.mode
    rows = [evaluate_pair(model, hw, eval_mode) for model in models for hw in hardware]
    comparisons = compare_bandwidths(rows) if args.compare_bandwidths else []
    recommendations = {
        "best_classic_16k": pick_best(rows, category="classic_mcu", sample_rate=16000, key_fn=row_sort_key_quality),
        "best_classic_8k": pick_best(rows, category="classic_mcu", sample_rate=8000, key_fn=row_sort_key_quality),
        "best_mcu_npu_16k": pick_best(rows, category="mcu_npu", sample_rate=16000, key_fn=row_sort_key_quality),
        "best_mcu_npu_8k": pick_best(rows, category="mcu_npu", sample_rate=8000, key_fn=row_sort_key_quality),
    }
    recommendations["quality_first"] = recommendation_block(rows, pick_overall(rows, row_sort_key_quality), row_sort_key_quality)
    recommendations["efficiency_first"] = recommendation_block(rows, pick_overall(rows, row_sort_key_efficiency), row_sort_key_efficiency)
    porting_target = build_porting_target(rows) if args.mode == "porting-target" else None

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "source": args.source,
        "eval_mode": eval_mode,
        "compare_bandwidths": args.compare_bandwidths,
        "rows": rows,
        "comparisons": comparisons,
        "recommendations": recommendations,
        "porting_target": porting_target,
    }

    out_json, out_csv, out_md = resolve_output_paths(args)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_csv, rows)
    out_md.write_text(build_markdown(rows, comparisons, recommendations, args.mode, porting_target), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
