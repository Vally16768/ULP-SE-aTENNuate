from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import mlflow
import torch
from mlflow import MlflowClient

from sebench.checkpoints import load_model_from_checkpoint
from sebench.classic_baselines import classic_pesq_index, normalize_baseline_name, summarize_classic_baselines
from sebench.mlflow_utils import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL,
    DEFAULT_TRACKING_URI,
    count_runs_by_status,
    configure_mlflow,
    find_finished_run,
    log_dict_artifact,
    register_run_model,
    terminate_matching_runs,
)
from sebench.audio import resample_mono_audio
from sebench.data import read_pair_manifest
from sebench.models import (
    METRICGAN_PLUS_SOURCE,
    MetricGANPlusAdapter,
    build_enhancer,
    build_metricgan_standalone,
    dynamic_quantize_metricgan,
)
from sebench.postfilters import resolve_postfilter_config, spectral_gate_waveform
from sebench.runtime import require_cuda_device
from sebench.splits import build_voicebank_campaign_splits
from sebench.stm32sim import (
    DEFAULT_MCU_REFERENCE_PROFILES,
    DEFAULT_MCU_SHORTLIST_PROFILES,
    parse_profile_names,
    simulate_baseline_across_profiles,
    simulate_classic_baseline,
    simulate_metricgan_plus_reference_across_profiles,
    simulate_model_across_profiles,
    simulate_model_fit,
)
from sebench.teacher_cache import build_teacher_cache
from sebench.training import (
    ExperimentConfig,
    benchmark_inference,
    evaluate_manifest,
    install_termination_handlers,
    restore_termination_handlers,
    run_experiment,
    summary_from_existing,
)


DEFAULT_GATING_EXPERIMENT_NAME = "voicebank-demand-pesq-spectral-gating"
DEFAULT_CASCADE_EXPERIMENT_NAME = "voicebank-demand-pesq-cascade"
DEFAULT_STM32_EXPERIMENT_NAME = "voicebank-demand-pesq-stm32"
DEFAULT_STM32_8K_EXPERIMENT_NAME = "voicebank-demand-pesq-stm32-8k"
DEFAULT_TEACHER_AUDIT_EXPERIMENT_NAME = "voicebank-demand-pesq-teacher-audit"
DEFAULT_TEACHER_LITE_EXPERIMENT_NAME = "voicebank-demand-pesq-teacher-lite"
DEFAULT_DEPLOY_WINNER_PESQ = 2.6559
LOW_POWER_PROFILE_NAMES = {
    "stm32u5_low_power_rt",
    "nrf54h20_low_power_rt",
    "apollo4_blue_plus_low_power_rt",
}
LARGE_MCU_DEMO_PROFILE_NAMES = {
    "imx_rt700_ai_audio_rt",
    "stm32n6_ai_audio_rt",
}


def campaign_log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[campaign {timestamp}] {message}", flush=True)


class SpectralGatingBaseline(torch.nn.Module):
    def __init__(self, *, n_fft: int, hop_length: int, win_length: int, preset: str = "medium") -> None:
        super().__init__()
        config = resolve_postfilter_config("sg_input_floor", preset)
        self.config = type(config)(
            mode=config.mode,
            preset=config.preset,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            percentile=config.percentile,
            percentile_window=config.percentile_window,
            freq_window=config.freq_window,
            time_window=config.time_window,
            strength=config.strength,
            threshold_scale=config.threshold_scale,
            temperature=config.temperature,
            min_mask=config.min_mask,
        )

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        return spectral_gate_waveform(noisy, noisy, self.config)

    def denoise_single(self, noisy: torch.Tensor) -> torch.Tensor:
        return self.forward(noisy)


class ResampledTeacherWrapper(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        input_sample_rate: int,
        model_sample_rate: int,
        output_sample_rate: int,
        output_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.input_sample_rate = input_sample_rate
        self.model_sample_rate = model_sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_device = output_device

    def denoise_single(self, noisy: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 2:
            raise ValueError("Expected noisy tensor shaped (batch, length).")
        resampled = resample_mono_audio(noisy, self.input_sample_rate, self.model_sample_rate)
        enhanced = self.base_model.denoise_single(resampled.to(self.output_device)).cpu()
        return resample_mono_audio(enhanced, self.model_sample_rate, self.output_sample_rate)


def resolve_stm32_audio_args(args: argparse.Namespace) -> None:
    default_train_csv = "/mnt/ldm/ULP-SE-aTENNuate/dataset/voicebank-demand/16k/train.csv"
    default_test_csv = "/mnt/ldm/ULP-SE-aTENNuate/dataset/voicebank-demand/16k/test.csv"
    default_splits_dir = "/mnt/ldm/ULP-SE-aTENNuate/dataset/voicebank-demand/16k/campaign"
    if args.stm32_audio_profile == "8k":
        defaults = {
            "sample_rate": 8000,
            "n_fft": 256,
            "hop_length": 80,
            "win_length": 160,
            "segment_len": 16000,
            "experiment_name": DEFAULT_STM32_8K_EXPERIMENT_NAME,
        }
        if args.train_csv == default_train_csv:
            candidate = default_train_csv.replace("/16k/", "/8k/")
            args.train_csv = candidate if Path(candidate).exists() else default_train_csv
        if args.test_csv == default_test_csv:
            candidate = default_test_csv.replace("/16k/", "/8k/")
            args.test_csv = candidate if Path(candidate).exists() else default_test_csv
        if args.splits_dir == default_splits_dir:
            args.splits_dir = default_splits_dir.replace("/16k/", "/8k/")
    else:
        defaults = {
            "sample_rate": 16000,
            "n_fft": 512,
            "hop_length": 160,
            "win_length": 320,
            "segment_len": 32000,
            "experiment_name": DEFAULT_STM32_EXPERIMENT_NAME,
        }
    if args.stm32_sample_rate is None:
        args.stm32_sample_rate = defaults["sample_rate"]
    if args.stm32_n_fft is None:
        args.stm32_n_fft = defaults["n_fft"]
    if args.stm32_hop_length is None:
        args.stm32_hop_length = defaults["hop_length"]
    if args.stm32_win_length is None:
        args.stm32_win_length = defaults["win_length"]
    args.stm32_segment_len = defaults["segment_len"]
    if args.stm32_experiment_name == DEFAULT_STM32_EXPERIMENT_NAME and args.stm32_audio_profile == "8k":
        args.stm32_experiment_name = defaults["experiment_name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VoiceBank+DEMAND PESQ benchmark campaign.")
    parser.add_argument("--train-csv", default="/mnt/ldm/ULP-SE-aTENNuate/dataset/voicebank-demand/16k/train.csv")
    parser.add_argument("--test-csv", default="/mnt/ldm/ULP-SE-aTENNuate/dataset/voicebank-demand/16k/test.csv")
    parser.add_argument("--splits-dir", default="/mnt/ldm/ULP-SE-aTENNuate/dataset/voicebank-demand/16k/campaign")
    parser.add_argument(
        "--phase",
        default="all",
        choices=[
            "all",
            "phase0",
            "phase1",
            "phase2",
            "phase3",
            "phase4",
            "gating_stage1",
            "gating_stage2",
            "gating_stage3",
            "gating_all",
            "cascade_stage1",
            "cascade_stage2",
            "cascade_expand",
            "cascade_test",
            "cascade_auto_next",
            "cascade_auto",
            "cascade_all",
            "stm32_teacher_cache",
            "stm32_classic_baseline",
            "stm32_stage0_sim",
            "stm32_stage1",
            "stm32_expand",
            "stm32_qat",
            "stm32_test",
            "stm32_auto",
            "teacher16k_fp32_ref",
            "teacher16k_int8_bench",
            "teacher8k_native_train",
            "teacher8k_native_int8_bench",
            "teacher_mcu_decision",
            "teacher_lite_stage0_sim",
            "teacher_lite_stage1_train",
            "teacher_lite_qat",
            "teacher_lite_decision",
        ],
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for debugging a subset of a phase.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mlflow-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--mlflow-artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--gating-experiment-name", default=DEFAULT_GATING_EXPERIMENT_NAME)
    parser.add_argument("--cascade-experiment-name", default=DEFAULT_CASCADE_EXPERIMENT_NAME)
    parser.add_argument("--stm32-experiment-name", default=DEFAULT_STM32_EXPERIMENT_NAME)
    parser.add_argument("--teacher-audit-experiment-name", default=DEFAULT_TEACHER_AUDIT_EXPERIMENT_NAME)
    parser.add_argument("--teacher-lite-experiment-name", default=DEFAULT_TEACHER_LITE_EXPERIMENT_NAME)
    parser.add_argument("--classic-baselines-xlsx", default="/home/vali/Desktop/comparison_with_classic_baselines.xlsx")
    parser.add_argument("--registered-model-name", default=DEFAULT_REGISTERED_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs-smoke", type=int, default=3)
    parser.add_argument("--epochs-phase1", type=int, default=20)
    parser.add_argument("--epochs-phase2", type=int, default=30)
    parser.add_argument("--epochs-phase3", type=int, default=30)
    parser.add_argument("--epochs-phase4", type=int, default=40)
    parser.add_argument("--epochs-gating-train", type=int, default=100)
    parser.add_argument("--gating-min-epochs", type=int, default=20)
    parser.add_argument("--gating-early-stop-patience", type=int, default=10)
    parser.add_argument("--epochs-cascade-train", type=int, default=100)
    parser.add_argument("--cascade-min-epochs", type=int, default=20)
    parser.add_argument("--cascade-early-stop-patience", type=int, default=10)
    parser.add_argument("--cascade-improve-threshold", type=float, default=0.02)
    parser.add_argument("--target-pesq", type=float, default=2.4)
    parser.add_argument("--auto-next-max-val-gap", type=float, default=0.1)
    parser.add_argument("--auto-next-max-extra-tests", type=int, default=2)
    parser.add_argument("--epochs-stm32-train", type=int, default=100)
    parser.add_argument("--epochs-stm32-qat", type=int, default=20)
    parser.add_argument("--epochs-teacher8k-train", type=int, default=40)
    parser.add_argument("--teacher8k-min-epochs", type=int, default=10)
    parser.add_argument("--teacher8k-early-stop-patience", type=int, default=5)
    parser.add_argument("--teacher8k-lr-patience", type=int, default=2)
    parser.add_argument("--stm32-min-epochs", type=int, default=15)
    parser.add_argument("--stm32-early-stop-patience", type=int, default=8)
    parser.add_argument("--stm32-lr-patience", type=int, default=3)
    parser.add_argument("--stm32-improve-threshold", type=float, default=0.05)
    parser.add_argument("--stm32-teacher-gap-max", type=float, default=0.15)
    parser.add_argument("--teacher-lite-target-pesq", type=float, default=DEFAULT_DEPLOY_WINNER_PESQ)
    parser.add_argument("--stm32-audio-profile", default="16k", choices=["16k", "8k"])
    parser.add_argument("--stm32-sample-rate", type=int, default=None)
    parser.add_argument("--stm32-n-fft", type=int, default=None)
    parser.add_argument("--stm32-hop-length", type=int, default=None)
    parser.add_argument("--stm32-win-length", type=int, default=None)
    parser.add_argument("--stm32-profile", default="imx_rt700_ai_audio_rt")
    parser.add_argument("--mcu-shortlist-profiles", default=",".join(DEFAULT_MCU_SHORTLIST_PROFILES))
    parser.add_argument("--mcu-reference-profiles", default=",".join(DEFAULT_MCU_REFERENCE_PROFILES))
    return parser.parse_args()


def score_value(result: dict[str, Any]) -> float:
    return float(result.get("best_val_select_pesq") or result.get("best_val_rank_pesq") or float("-inf"))


def _stm32_deploy_summary(result: dict[str, Any]) -> dict[str, Any]:
    return _stm32_deploy_summary_mode(result, require_power=True)


def _pick_best_profile_from_names(profiles: dict[str, Any], names: list[str]) -> dict[str, Any]:
    best_name = None
    best_power = None
    best_mhz = None
    for name in names:
        summary = profiles.get(name)
        if summary is None:
            continue
        avg_power = float(summary.get("avg_power_mw_at_recommended_mhz") or summary.get("avg_power_mw") or float("inf"))
        required_mhz = float(summary.get("recommended_rt_mhz") or summary.get("min_required_mhz") or float("inf"))
        if (
            best_name is None
            or avg_power < best_power
            or (avg_power == best_power and required_mhz < best_mhz)
        ):
            best_name = name
            best_power = avg_power
            best_mhz = required_mhz
    return {
        "profile_name": best_name,
        "avg_power_mw": best_power,
        "recommended_rt_mhz": best_mhz,
        "realtime_ok": bool(best_name),
    }


def _stm32_deploy_summary_mode(result: dict[str, Any], *, require_power: bool) -> dict[str, Any]:
    shortlist = (result.get("mcu_shortlist") or {}) if isinstance(result, dict) else {}
    if shortlist:
        profile_key = "power_supported_profiles" if require_power else "hardware_supported_profiles"
        supported_names = list(shortlist.get(profile_key) or [])
        profiles = shortlist.get("profiles") or {}
        if supported_names and profiles:
            return _pick_best_profile_from_names(profiles, supported_names)
        if require_power:
            return {
                "profile_name": shortlist.get("best_power_profile_name") or shortlist.get("best_profile_name"),
                "avg_power_mw": shortlist.get("best_power_profile_avg_power_mw") or shortlist.get("lowest_avg_power_mw"),
                "recommended_rt_mhz": shortlist.get("best_power_profile_recommended_rt_mhz") or shortlist.get("lowest_required_mhz"),
                "realtime_ok": int(shortlist.get("power_supported_profile_count") or 0) > 0,
            }
        return {
            "profile_name": shortlist.get("lowest_avg_power_profile_name") or shortlist.get("best_profile_name"),
            "avg_power_mw": shortlist.get("lowest_avg_power_mw"),
            "recommended_rt_mhz": shortlist.get("lowest_required_mhz"),
            "realtime_ok": int(shortlist.get("hardware_supported_profile_count") or 0) > 0,
        }
    stm32sim = (result.get("stm32sim") or {}) if isinstance(result, dict) else {}
    return {
        "profile_name": result.get("mcu_profile") or stm32sim.get("profile_name"),
        "avg_power_mw": stm32sim.get("avg_power_mw_at_recommended_mhz") or stm32sim.get("avg_power_mw"),
        "recommended_rt_mhz": stm32sim.get("recommended_rt_mhz") or stm32sim.get("min_required_mhz"),
        "realtime_ok": (
            bool(stm32sim.get("fit_ok"))
            and bool(stm32sim.get("frequency_ok"))
            and bool(stm32sim.get("realtime_ok"))
            and (bool(stm32sim.get("power_ok")) if require_power else True)
            and (bool(stm32sim.get("latency_ok")) if "latency_ok" in stm32sim else True)
        ),
    }


def _stm32_candidate_sort_key(result: dict[str, Any], *, require_power: bool = True) -> tuple[float, float, float]:
    deploy = _stm32_deploy_summary_mode(result, require_power=require_power)
    pesq = score_value(result)
    avg_power_mw = float(deploy.get("avg_power_mw") or float("inf"))
    required_mhz = float(deploy.get("recommended_rt_mhz") or float("inf"))
    return (pesq, -avg_power_mw, -required_mhz)


def _with_stm32_recommendation(result: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(result)
    deploy = _stm32_deploy_summary_mode(result, require_power=True)
    hardware_deploy = _stm32_deploy_summary_mode(result, require_power=False)
    enriched["deployment_profile_name"] = deploy.get("profile_name")
    enriched["deployment_avg_power_mw"] = deploy.get("avg_power_mw")
    enriched["deployment_recommended_rt_mhz"] = deploy.get("recommended_rt_mhz")
    enriched["deployment_realtime_ok"] = deploy.get("realtime_ok")
    enriched["hardware_deployment_profile_name"] = hardware_deploy.get("profile_name")
    enriched["hardware_deployment_avg_power_mw"] = hardware_deploy.get("avg_power_mw")
    enriched["hardware_deployment_recommended_rt_mhz"] = hardware_deploy.get("recommended_rt_mhz")
    enriched["hardware_deployment_realtime_ok"] = hardware_deploy.get("realtime_ok")
    return enriched


def run_name(config: ExperimentConfig) -> str:
    return (
        f"{config.model_family}-{config.variant}-lr{config.lr:g}-seg{config.segment_len}"
        f"-loss{config.loss_recipe}-seed{config.seed}"
    )


@dataclass(frozen=True)
class PostfilterEvalSpec:
    source_run_id: str
    source_run_name: str
    checkpoint_out: str
    model_family: str
    variant: str
    loss_recipe: str
    scheduler: str | None
    lr: float
    segment_len: int
    seed: int | None
    phase: str
    val_select_csv: str
    mlflow_uri: str
    mlflow_artifact_root: str
    experiment_name: str
    device: str
    benchmark_seconds: int = 10
    benchmark_repeats: int = 3
    sample_count: int = 3
    postfilter_mode: str = "none"
    postfilter_preset: str = "medium"
    train_postfilter: bool = False
    spectral_native_gate: bool = False

    @property
    def run_name(self) -> str:
        preset = "off" if self.postfilter_mode == "none" else self.postfilter_preset
        return f"{self.source_run_name}-pf{self.postfilter_mode}-{preset}"


@dataclass(frozen=True)
class CascadeStage1Spec:
    model_family: str
    variant: str
    phase: str
    val_select_csv: str
    mlflow_uri: str
    mlflow_artifact_root: str
    experiment_name: str
    device: str
    benchmark_seconds: int = 10
    benchmark_repeats: int = 1
    sample_count: int = 3
    postfilter_mode: str = "none"
    postfilter_preset: str = "medium"

    @property
    def run_name(self) -> str:
        preset = "off" if self.postfilter_mode == "none" else self.postfilter_preset
        return f"{self.model_family}-{self.variant}-pretrained-pf{self.postfilter_mode}-{preset}"


def stm32_candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "model_family": "tiny_stm32_fc",
            "variant": "small",
            "guidance_classic": "none",
            "loss_recipe": "D1",
        },
        {
            "model_family": "tiny_stm32_hybrid_sg",
            "variant": "small",
            "guidance_classic": "spectral_gating",
            "loss_recipe": "D1",
        },
        {
            "model_family": "tiny_stm32_tcn_hybrid",
            "variant": "small",
            "guidance_classic": "spectral_gating",
            "loss_recipe": "D1",
        },
    ]


def teacher_lite_candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "model_family": "metricgan_plus_native8k_causal_s",
            "variant": "small",
            "loss_recipe": "D1",
        },
        {
            "model_family": "metricgan_plus_native8k_causal_xs",
            "variant": "small",
            "loss_recipe": "D1",
        },
        {
            "model_family": "metricgan_plus_native8k_causal_n6",
            "variant": "small",
            "loss_recipe": "D1",
        },
    ]


def _stm32_metrics_from_summary(result: dict[str, Any]) -> dict[str, float | None]:
    stm32sim = result.get("stm32sim") or {}
    return {
        "flash_bytes": stm32sim.get("flash_bytes"),
        "sram_peak_bytes": stm32sim.get("sram_peak_bytes"),
        "cycles_per_hop": stm32sim.get("cycles_per_hop"),
        "ms_per_hop_80mhz": stm32sim.get("ms_per_hop_80mhz"),
        "cpu_load_pct": stm32sim.get("cpu_load_pct"),
        "fit_ok": stm32sim.get("fit_ok"),
        "realtime_ok": stm32sim.get("realtime_ok"),
    }


def _stm32_sim_run_name(spec: dict[str, Any]) -> str:
    return f"{spec['model_family']}-{spec['variant']}-sim-{spec['guidance_classic']}"


def _teacher_lite_sim_run_name(spec: dict[str, Any]) -> str:
    return f"{spec['model_family']}-{spec['variant']}-sim"


def _search_best_teacher_run(args: argparse.Namespace) -> dict[str, Any] | None:
    client = MlflowClient(tracking_uri=args.mlflow_uri)
    experiment = client.get_experiment_by_name(args.cascade_experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED' and tags.phase = 'cascade_test'",
        order_by=["metrics.`test/pesq_mean` DESC"],
        max_results=20,
    )
    for run in runs:
        name = run.data.tags.get("mlflow.runName") or ""
        if "metricgan_plus" not in name:
            continue
        return {
            "run_id": run.info.run_id,
            "run_name": name,
            "test_pesq": run.data.metrics.get("test/pesq_mean"),
        }
    return None


def _find_named_run(
    tracking_uri: str,
    experiment_name: str,
    *,
    run_name: str,
    phase: str | None = None,
) -> dict[str, Any] | None:
    return find_finished_run(tracking_uri, experiment_name, run_name, phase=phase)


def _teacher_lite_teacher_checkpoint_path(args: argparse.Namespace) -> tuple[Path, dict[str, Any] | None]:
    checkpoint_path = Path("checkpoints/teacher_audit/metricgan_plus_native8k_small.pt")
    existing = _find_named_run(
        args.mlflow_uri,
        args.teacher_audit_experiment_name,
        run_name="metricgan_plus_native8k-small-fp32",
        phase="teacher8k_native_train",
    )
    if existing:
        checkpoint_out = existing.get("params", {}).get("checkpoint_out")
        if checkpoint_out:
            candidate = Path(str(checkpoint_out))
            if candidate.exists():
                return candidate, teacher_audit_result_from_existing(existing)
    if checkpoint_path.exists():
        return checkpoint_path, (
            teacher_audit_result_from_existing(existing) if existing is not None else None
        )
    raise FileNotFoundError(
        "Teacher-lite source checkpoint not found. Expected teacher audit checkpoint "
        f"at {checkpoint_path} or a finished teacher8k_native_train run in {args.teacher_audit_experiment_name}."
    )


def _teacher_lite_teacher_int8_run(args: argparse.Namespace) -> dict[str, Any] | None:
    existing = _find_named_run(
        args.mlflow_uri,
        args.teacher_audit_experiment_name,
        run_name="metricgan_plus_native8k-small-int8",
        phase="teacher8k_native_int8_bench",
    )
    if existing:
        return teacher_audit_result_from_existing(existing)
    return None


def _load_teacher_lite_source_model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any] | None]:
    checkpoint_path, source_summary = _teacher_lite_teacher_checkpoint_path(args)
    model, _ = load_model_from_checkpoint(
        checkpoint_path,
        device="cpu",
        model_family="metricgan_plus_native8k",
        variant="small",
    )
    return dynamic_quantize_metricgan(model), source_summary


def _search_current_stm32_classic_result(args: argparse.Namespace) -> dict[str, Any]:
    existing = _find_named_run(
        args.mlflow_uri,
        args.stm32_experiment_name,
        run_name="stm32_classic_baseline",
        phase="stm32_classic_baseline",
    )
    if existing:
        return stm32_result_from_existing(existing)
    return {
        "run_name": "stm32_classic_baseline",
        "best_val_select_pesq": 2.3762 if args.stm32_sample_rate <= 8000 else 2.2139,
        "mcu_shortlist": {},
        "stm32sim": {},
    }


def _search_current_deploy_winner(args: argparse.Namespace, *, require_power: bool = True) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for phase in ("stm32_test", "stm32_qat", "stm32_expand", "stm32_stage1"):
        candidates.extend(search_finished_results(args.mlflow_uri, args.stm32_experiment_name, phase=phase))
    valid = [
        _with_stm32_recommendation(result)
        for result in candidates
        if result.get("model_family") in {"tiny_stm32_fc", "tiny_stm32_hybrid_sg", "tiny_stm32_tcn_hybrid", "spectral_gating"}
        and _stm32_sim_is_eligible(result, require_power=require_power)
        and result.get("best_val_select_pesq") is not None
    ]
    if not valid:
        return {
            "run_name": "tiny_stm32_hybrid_sg-default",
            "model_family": "tiny_stm32_hybrid_sg",
            "best_val_select_pesq": DEFAULT_DEPLOY_WINNER_PESQ,
            "mcu_shortlist": {},
            "stm32sim": {},
            "deployment_profile_name": None,
            "deployment_avg_power_mw": None,
            "deployment_recommended_rt_mhz": None,
            "hardware_deployment_profile_name": None,
            "hardware_deployment_avg_power_mw": None,
            "hardware_deployment_recommended_rt_mhz": None,
        }
    return max(valid, key=lambda item: _stm32_candidate_sort_key(item, require_power=require_power))


def mcu_shortlist_profiles(args: argparse.Namespace) -> tuple[str, ...]:
    return parse_profile_names(args.mcu_shortlist_profiles, DEFAULT_MCU_SHORTLIST_PROFILES)


def mcu_reference_profiles(args: argparse.Namespace) -> tuple[str, ...]:
    return parse_profile_names(args.mcu_reference_profiles, DEFAULT_MCU_REFERENCE_PROFILES)


def _stm32_metric_fields(summary: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "stm32sim/flash_bytes": float(summary["flash_bytes"]),
        "stm32sim/sram_peak_bytes": float(summary["sram_peak_bytes"]),
        "stm32sim/macs_per_hop_total": float(summary["macs_per_hop_total"]),
        "stm32sim/macs_fc": float(summary["macs_fc"]),
        "stm32sim/macs_depthwise_conv1d": float(summary["macs_depthwise_conv1d"]),
        "stm32sim/macs_pointwise_conv1d": float(summary["macs_pointwise_conv1d"]),
        "stm32sim/macs_lstm": float(summary["macs_lstm"]),
        "stm32sim/eltwise_ops": float(summary["eltwise_ops"]),
        "stm32sim/lookup_ops": float(summary["lookup_ops"]),
        "stm32sim/cycles_per_hop": float(summary["cycles_per_hop"]),
        "stm32sim/ms_per_hop_80mhz": float(summary["ms_per_hop_80mhz"]),
        "stm32sim/hop_ms": float(summary["hop_ms"]),
        "stm32sim/lookahead_ms": float(summary.get("lookahead_ms") or 0.0),
        "stm32sim/min_required_mhz": float(summary["min_required_mhz"]),
        "stm32sim/recommended_rt_mhz": float(summary["recommended_rt_mhz"]),
        "stm32sim/max_profile_mhz": float(summary["max_profile_mhz"]),
        "stm32sim/cpu_load_pct": float(summary["cpu_load_pct"]),
        "stm32sim/fit_ok": 1.0 if summary["fit_ok"] else 0.0,
        "stm32sim/frequency_ok": 1.0 if summary["frequency_ok"] else 0.0,
        "stm32sim/realtime_ok": 1.0 if summary["realtime_ok"] else 0.0,
        "stm32sim/latency_ok": 1.0 if summary.get("latency_ok") else 0.0,
    }
    if "avg_power_mw" in summary:
        metrics["stm32sim/avg_power_mw"] = float(summary["avg_power_mw"])
    if "avg_power_mw_at_recommended_mhz" in summary:
        metrics["stm32sim/avg_power_mw_at_recommended_mhz"] = float(summary["avg_power_mw_at_recommended_mhz"])
    if "energy_uj_per_hop" in summary:
        metrics["stm32sim/energy_uj_per_hop"] = float(summary["energy_uj_per_hop"])
    if "energy_uj_per_hop_at_recommended_mhz" in summary:
        metrics["stm32sim/energy_uj_per_hop_at_recommended_mhz"] = float(summary["energy_uj_per_hop_at_recommended_mhz"])
    if "power_ok" in summary:
        metrics["stm32sim/power_ok"] = 1.0 if summary["power_ok"] else 0.0
    if "deployment_ok" in summary:
        metrics["stm32sim/deployment_ok"] = 1.0 if summary["deployment_ok"] else 0.0
    return metrics


def _mcu_shortlist_has_power_fields(shortlist: dict[str, Any] | None) -> bool:
    if not shortlist:
        return False
    return (
        "power_supported_profile_count" in shortlist
        and "low_power_supported_profile_count" in shortlist
        and "best_power_profile_recommended_rt_mhz" in shortlist
        and "lowest_required_mhz" in shortlist
        and shortlist.get("lowest_required_mhz") is not None
    )


def _stm32_summary_has_frequency_fields(summary: dict[str, Any] | None) -> bool:
    if not summary:
        return False
    return summary.get("min_required_mhz") is not None and summary.get("recommended_rt_mhz") is not None


def _mcu_shortlist_summary_from_raw(metrics: dict[str, Any], tags: dict[str, Any]) -> dict[str, Any] | None:
    if "mcu_shortlist/supported_profile_count" not in metrics and "mcu_shortlist/supported_profiles" not in tags:
        return None
    best_ms = metrics.get("mcu_shortlist/best_ms_per_hop_profile")
    best_power = metrics.get("mcu_shortlist/best_power_profile_avg_power_mw")
    best_power_mhz = metrics.get("mcu_shortlist/best_power_profile_recommended_rt_mhz")
    lowest_power = metrics.get("mcu_shortlist/lowest_avg_power_mw")
    lowest_mhz = metrics.get("mcu_shortlist/lowest_required_mhz")
    return {
        "supported_profile_count": int(metrics.get("mcu_shortlist/supported_profile_count") or 0),
        "hardware_supported_profile_count": int(metrics.get("mcu_shortlist/hardware_supported_profile_count") or 0),
        "reference_supported_profile_count": int(metrics.get("mcu_shortlist/reference_supported_profile_count") or 0),
        "power_supported_profile_count": int(metrics.get("mcu_shortlist/power_supported_profile_count") or 0),
        "low_power_supported_profile_count": int(metrics.get("mcu_shortlist/low_power_supported_profile_count") or 0),
        "supported_profiles": [item for item in str(tags.get("mcu_shortlist/supported_profiles") or "").split(",") if item],
        "hardware_supported_profiles": [
            item for item in str(tags.get("mcu_shortlist/hardware_supported_profiles") or "").split(",") if item
        ],
        "reference_supported_profiles": [
            item for item in str(tags.get("mcu_shortlist/reference_supported_profiles") or "").split(",") if item
        ],
        "power_supported_profiles": [
            item for item in str(tags.get("mcu_shortlist/power_supported_profiles") or "").split(",") if item
        ],
        "low_power_supported_profiles": [
            item for item in str(tags.get("mcu_shortlist/low_power_supported_profiles") or "").split(",") if item
        ],
        "best_profile_name": tags.get("mcu_shortlist/best_profile_name"),
        "best_power_profile_name": tags.get("mcu_shortlist/best_power_profile_name"),
        "lowest_required_mhz_profile_name": tags.get("mcu_shortlist/lowest_required_mhz_profile_name"),
        "lowest_avg_power_profile_name": tags.get("mcu_shortlist/lowest_avg_power_profile_name"),
        "best_ms_per_hop_profile": float(best_ms) if best_ms is not None else None,
        "best_power_profile_avg_power_mw": float(best_power) if best_power is not None else None,
        "best_power_profile_recommended_rt_mhz": float(best_power_mhz) if best_power_mhz is not None else None,
        "lowest_avg_power_mw": float(lowest_power) if lowest_power is not None else None,
        "lowest_required_mhz": float(lowest_mhz) if lowest_mhz is not None else None,
    }


def _shortlist_metrics_from_audit(audit: dict[str, Any]) -> dict[str, float]:
    best_ms = audit.get("best_ms_per_hop_profile")
    best_power = audit.get("best_power_profile_avg_power_mw")
    best_power_mhz = audit.get("best_power_profile_recommended_rt_mhz")
    lowest_power = audit.get("lowest_avg_power_mw")
    lowest_mhz = audit.get("lowest_required_mhz")
    metrics: dict[str, float] = {
        "mcu_shortlist/supported_profile_count": float(audit.get("supported_profile_count") or 0),
        "mcu_shortlist/hardware_supported_profile_count": float(audit.get("hardware_supported_profile_count") or 0),
        "mcu_shortlist/reference_supported_profile_count": float(audit.get("reference_supported_profile_count") or 0),
        "mcu_shortlist/power_supported_profile_count": float(audit.get("power_supported_profile_count") or 0),
        "mcu_shortlist/low_power_supported_profile_count": float(audit.get("low_power_supported_profile_count") or 0),
    }
    if best_ms is not None:
        metrics["mcu_shortlist/best_ms_per_hop_profile"] = float(best_ms)
    if best_power is not None:
        metrics["mcu_shortlist/best_power_profile_avg_power_mw"] = float(best_power)
    if best_power_mhz is not None:
        metrics["mcu_shortlist/best_power_profile_recommended_rt_mhz"] = float(best_power_mhz)
    if lowest_power is not None:
        metrics["mcu_shortlist/lowest_avg_power_mw"] = float(lowest_power)
    if lowest_mhz is not None:
        metrics["mcu_shortlist/lowest_required_mhz"] = float(lowest_mhz)
    return metrics


def _log_shortlist_audit_to_run(
    tracking_uri: str,
    run_id: str,
    audit: dict[str, Any],
    *,
    artifact_name: str = "mcu_shortlist.json",
) -> None:
    client = MlflowClient(tracking_uri=tracking_uri)
    for key, value in _shortlist_metrics_from_audit(audit).items():
        client.log_metric(run_id, key, value)
    client.set_tag(run_id, "mcu_shortlist/supported_profiles", ",".join(audit.get("supported_profiles") or []))
    client.set_tag(
        run_id,
        "mcu_shortlist/hardware_supported_profiles",
        ",".join(audit.get("hardware_supported_profiles") or []),
    )
    client.set_tag(
        run_id,
        "mcu_shortlist/reference_supported_profiles",
        ",".join(audit.get("reference_supported_profiles") or []),
    )
    client.set_tag(run_id, "mcu_shortlist/power_supported_profiles", ",".join(audit.get("power_supported_profiles") or []))
    client.set_tag(
        run_id,
        "mcu_shortlist/low_power_supported_profiles",
        ",".join(audit.get("low_power_supported_profiles") or []),
    )
    if audit.get("best_profile_name"):
        client.set_tag(run_id, "mcu_shortlist/best_profile_name", str(audit["best_profile_name"]))
    if audit.get("best_power_profile_name"):
        client.set_tag(run_id, "mcu_shortlist/best_power_profile_name", str(audit["best_power_profile_name"]))
    if audit.get("lowest_required_mhz_profile_name"):
        client.set_tag(run_id, "mcu_shortlist/lowest_required_mhz_profile_name", str(audit["lowest_required_mhz_profile_name"]))
    if audit.get("lowest_avg_power_profile_name"):
        client.set_tag(run_id, "mcu_shortlist/lowest_avg_power_profile_name", str(audit["lowest_avg_power_profile_name"]))
    with tempfile.TemporaryDirectory(prefix="mcu_shortlist_") as tmpdir:
        payload_path = Path(tmpdir) / artifact_name
        payload_path.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str), encoding="utf-8")
        client.log_artifact(run_id, str(payload_path), artifact_path="reports")


def run_mcu_teacher_audit(args: argparse.Namespace, teacher_ref: dict[str, Any] | None) -> dict[str, Any]:
    int8_audit = simulate_metricgan_plus_reference_across_profiles(
        shortlist_profiles=mcu_shortlist_profiles(args),
        reference_profiles=mcu_reference_profiles(args),
        weight_bits=8,
    )
    fp32_audit = simulate_metricgan_plus_reference_across_profiles(
        shortlist_profiles=mcu_shortlist_profiles(args),
        reference_profiles=mcu_reference_profiles(args),
        weight_bits=32,
    )
    return {
        "teacher_source_run_id": teacher_ref["run_id"] if teacher_ref else None,
        "teacher_source_run_name": teacher_ref["run_name"] if teacher_ref else "metricgan_plus-small-pretrained",
        "teacher_test_pesq": teacher_ref.get("test_pesq") if teacher_ref else None,
        "int8": int8_audit,
        "fp32": fp32_audit,
        "direct_viable_shortlist_profiles": list(int8_audit.get("power_supported_profiles") or []),
        "direct_hardware_supported_shortlist_profiles": list(int8_audit.get("hardware_supported_profiles") or []),
        "direct_power_supported_shortlist_profiles": list(int8_audit.get("power_supported_profiles") or []),
        "direct_low_power_supported_shortlist_profiles": list(int8_audit.get("low_power_supported_profiles") or []),
    }


def teacher_audit_result_from_existing(existing: dict[str, Any]) -> dict[str, Any]:
    summary = summary_from_existing(existing)
    metrics = existing.get("metrics", {})
    params = existing.get("params", {})
    summary.update(
        {
            "teacher_variant": params.get("teacher_variant"),
            "audit_only": params.get("audit_only"),
            "quantize_dynamic": params.get("quantize_dynamic"),
            "teacher_accuracy_drop_pesq": metrics.get("teacher_accuracy_drop_pesq"),
            "teacher_accuracy_drop_stoi": metrics.get("teacher_accuracy_drop_stoi"),
            "teacher_accuracy_drop_sisdr": metrics.get("teacher_accuracy_drop_sisdr"),
            "quantization_drop_pesq": metrics.get("quantization_drop_pesq"),
            "quantization_drop_stoi": metrics.get("quantization_drop_stoi"),
            "quantization_drop_sisdr": metrics.get("quantization_drop_sisdr"),
            "proxy8k_val_select_pesq": metrics.get("teacher8kproxy/val_select_pesq_mean"),
            "proxy8k_val_select_stoi": metrics.get("teacher8kproxy/val_select_stoi_mean"),
            "proxy8k_val_select_sisdr": metrics.get("teacher8kproxy/val_select_sisdr_mean"),
            "proxy8k_test_pesq": metrics.get("teacher8kproxy/test_pesq_mean"),
            "proxy8k_test_stoi": metrics.get("teacher8kproxy/test_stoi_mean"),
            "proxy8k_test_sisdr": metrics.get("teacher8kproxy/test_sisdr_mean"),
        }
    )
    shortlist = _mcu_shortlist_summary_from_raw(metrics, existing.get("tags", {}))
    if shortlist is not None:
        summary["mcu_shortlist"] = shortlist
    return summary


def _teacher_reference_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    train_csv = str(args.train_csv)
    test_csv = str(args.test_csv)
    splits_dir = str(args.splits_dir)
    for profile_from, profile_to in (("/8k/", "/16k/"),):
        if profile_from in train_csv:
            train_csv = train_csv.replace(profile_from, profile_to)
        if profile_from in test_csv:
            test_csv = test_csv.replace(profile_from, profile_to)
        if profile_from in splits_dir:
            splits_dir = splits_dir.replace(profile_from, profile_to)
    return train_csv, test_csv, splits_dir


def _teacher_metric_delta(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    prefix: str = "teacher_accuracy_drop",
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    pairs = (
        ("pesq", "best_val_select_pesq"),
        ("stoi", "best_val_select_stoi"),
        ("sisdr", "test_metrics"),
    )
    ref_pesq = reference.get("best_val_select_pesq")
    cand_pesq = candidate.get("best_val_select_pesq")
    if ref_pesq is not None and cand_pesq is not None:
        deltas[f"{prefix}_pesq"] = float(ref_pesq) - float(cand_pesq)
    ref_stoi = reference.get("best_val_select_stoi")
    cand_stoi = candidate.get("best_val_select_stoi")
    if ref_stoi is not None and cand_stoi is not None:
        deltas[f"{prefix}_stoi"] = float(ref_stoi) - float(cand_stoi)
    ref_test = reference.get("test_metrics") or {}
    cand_test = candidate.get("test_metrics") or {}
    ref_sisdr = ref_test.get("sisdr_mean")
    cand_sisdr = cand_test.get("sisdr_mean")
    if ref_sisdr is not None and cand_sisdr is not None:
        deltas[f"{prefix}_sisdr"] = float(ref_sisdr) - float(cand_sisdr)
    return deltas


def _teacher_proxy_from_ref(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_val_select_pesq": reference.get("proxy8k_val_select_pesq"),
        "best_val_select_stoi": reference.get("proxy8k_val_select_stoi"),
        "test_metrics": {
            "pesq_mean": reference.get("proxy8k_test_pesq"),
            "stoi_mean": reference.get("proxy8k_test_stoi"),
            "sisdr_mean": reference.get("proxy8k_test_sisdr"),
        },
    }


def _teacher_candidate_sort_key(result: dict[str, Any]) -> tuple[float, float, float]:
    accuracy_drop = float(result.get("teacher_accuracy_drop_pesq") or float("inf"))
    deploy = _stm32_deploy_summary(result)
    avg_power_mw = float(deploy.get("avg_power_mw") or float("inf"))
    required_mhz = float(deploy.get("recommended_rt_mhz") or float("inf"))
    return (-accuracy_drop, -avg_power_mw, -required_mhz)


def _teacher_summary_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "best/val_select_pesq_mean": float(metrics["pesq_mean"]),
        "best/val_select_stoi_mean": float(metrics["stoi_mean"]),
        "best/val_select_sisdr_mean": float(metrics["sisdr_mean"]),
        "best/val_select_delta_snr_mean": float(metrics["delta_snr_mean"]),
    }


def _test_metric_fields(metrics: dict[str, Any]) -> dict[str, float]:
    output = {
        "test/pesq_mean": float(metrics["pesq_mean"]),
        "test/stoi_mean": float(metrics["stoi_mean"]),
        "test/sisdr_mean": float(metrics["sisdr_mean"]),
        "test/delta_snr_mean": float(metrics["delta_snr_mean"]),
    }
    if "csig_mean" in metrics:
        output["test/csig_mean"] = float(metrics["csig_mean"])
        output["test/cbak_mean"] = float(metrics["cbak_mean"])
        output["test/covl_mean"] = float(metrics["covl_mean"])
    return output


def _teacher_run_summary(
    *,
    run_id: str,
    run_name_value: str,
    teacher_variant: str,
    checkpoint_out: str | None,
    best_val_select: dict[str, Any],
    test_metrics: dict[str, Any],
    stm32sim: dict[str, Any],
    mcu_shortlist: dict[str, Any],
    quantize_dynamic: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "run_id": run_id,
        "run_name": run_name_value,
        "model_family": "metricgan_plus" if "native8k" not in teacher_variant else "metricgan_plus_native8k",
        "variant": "small",
        "phase": None,
        "checkpoint_out": checkpoint_out,
        "teacher_variant": teacher_variant,
        "audit_only": True,
        "quantize_dynamic": quantize_dynamic,
        "best_val_select_pesq": best_val_select.get("pesq_mean"),
        "best_val_select_stoi": best_val_select.get("stoi_mean"),
        "best_val_select_sisdr": best_val_select.get("sisdr_mean"),
        "test_metrics": test_metrics,
        "stm32sim": stm32sim,
        "mcu_shortlist": mcu_shortlist,
    }
    if extra:
        summary.update(extra)
    return summary


def _run_teacher_eval(
    *,
    args: argparse.Namespace,
    phase: str,
    run_name_value: str,
    model: torch.nn.Module,
    eval_device: str,
    eval_sample_rate: int,
    val_select_csv: str,
    test_csv: str,
    teacher_variant: str,
    quantize_dynamic_flag: bool,
    stm32_weight_bits: int,
    sim_model: torch.nn.Module | None = None,
    checkpoint_out: str | None = None,
    proxy_metrics: dict[str, Any] | None = None,
    teacher_ref_result: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    existing = find_finished_run(
        args.mlflow_uri,
        args.teacher_audit_experiment_name,
        run_name_value,
        phase=phase,
    )
    if existing and args.resume:
        return teacher_audit_result_from_existing(existing)

    experiment_id = configure_mlflow(args.mlflow_uri, args.teacher_audit_experiment_name, args.mlflow_artifact_root)
    terminate_matching_runs(
        tracking_uri=args.mlflow_uri,
        experiment_name=args.teacher_audit_experiment_name,
        run_name=run_name_value,
        phase=phase,
    )
    run = mlflow.start_run(
        run_name=run_name_value,
        experiment_id=experiment_id,
        tags={"phase": phase, "run_type": "child", "audit_only": "true"},
    )
    run_status = "FINISHED"
    try:
        sample_rows = read_pair_manifest(val_select_csv)
        sample_path = sample_rows[0].noisy
        audit_model = sim_model or model
        stm32sim = simulate_model_fit(audit_model, profile_name=args.stm32_profile, weight_bits=stm32_weight_bits)
        shortlist_audit = simulate_model_across_profiles(
            audit_model,
            shortlist_profiles=mcu_shortlist_profiles(args),
            reference_profiles=mcu_reference_profiles(args),
            weight_bits=stm32_weight_bits,
        )
        val_metrics = evaluate_manifest(
            model,
            val_select_csv,
            eval_device,
            sample_rate=eval_sample_rate,
            compute_dnsmos=False,
            compute_composite=False,
            batch_size=4,
            cache_audio=True,
            progress_callback=lambda message: campaign_log(f"{phase}: {run_name_value}: {message}"),
        )
        test_metrics = evaluate_manifest(
            model,
            test_csv,
            eval_device,
            sample_rate=eval_sample_rate,
            compute_dnsmos=False,
            compute_composite=False,
            batch_size=4,
            cache_audio=True,
            progress_callback=lambda message: campaign_log(f"{phase}: {run_name_value}: {message}"),
        )
        latency_seconds = benchmark_inference(
            model,
            sample_path=sample_path,
            device=eval_device,
            sample_rate=eval_sample_rate,
            duration_seconds=10,
            repeats=1,
        )
        params = {
            "model_family": "metricgan_plus" if "native8k" not in teacher_variant else "metricgan_plus_native8k",
            "variant": "small",
            "phase": phase,
            "teacher_variant": teacher_variant,
            "audit_only": True,
            "quantize_dynamic": quantize_dynamic_flag,
            "sample_rate": eval_sample_rate,
            "pretrained_source": METRICGAN_PLUS_SOURCE,
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)
        metrics = {
            **_teacher_summary_metrics(val_metrics),
            **_test_metric_fields(test_metrics),
            "best/inference_seconds_10s": latency_seconds,
            **_stm32_metric_fields(stm32sim),
            **_shortlist_metrics_from_audit(shortlist_audit),
        }
        if proxy_metrics:
            if proxy_metrics.get("val_select") is not None:
                metrics["teacher8kproxy/val_select_pesq_mean"] = float(proxy_metrics["val_select"]["pesq_mean"])
                metrics["teacher8kproxy/val_select_stoi_mean"] = float(proxy_metrics["val_select"]["stoi_mean"])
                metrics["teacher8kproxy/val_select_sisdr_mean"] = float(proxy_metrics["val_select"]["sisdr_mean"])
            if proxy_metrics.get("test") is not None:
                metrics["teacher8kproxy/test_pesq_mean"] = float(proxy_metrics["test"]["pesq_mean"])
                metrics["teacher8kproxy/test_stoi_mean"] = float(proxy_metrics["test"]["stoi_mean"])
                metrics["teacher8kproxy/test_sisdr_mean"] = float(proxy_metrics["test"]["sisdr_mean"])
        if teacher_ref_result is not None:
            deltas = _teacher_metric_delta(teacher_ref_result, {"best_val_select_pesq": val_metrics["pesq_mean"], "best_val_select_stoi": val_metrics["stoi_mean"], "test_metrics": test_metrics})
            if "teacher_accuracy_drop_pesq" in deltas:
                metrics["teacher_accuracy_drop_pesq"] = deltas["teacher_accuracy_drop_pesq"]
            if "teacher_accuracy_drop_stoi" in deltas:
                metrics["teacher_accuracy_drop_stoi"] = deltas["teacher_accuracy_drop_stoi"]
            if "teacher_accuracy_drop_sisdr" in deltas:
                metrics["teacher_accuracy_drop_sisdr"] = deltas["teacher_accuracy_drop_sisdr"]
        if extra_metrics:
            metrics.update(extra_metrics)
        mlflow.log_metrics(metrics)
        _log_shortlist_audit_to_run(args.mlflow_uri, run.info.run_id, shortlist_audit, artifact_name=f"{phase}_mcu_shortlist.json")
        mlflow.set_tag("audit_only", "true")
        log_dict_artifact(
            {
                "val_select_metrics": val_metrics,
                "test_metrics": test_metrics,
                "stm32sim": stm32sim,
                "mcu_shortlist": shortlist_audit,
                "proxy8k_metrics": proxy_metrics,
            },
            f"reports/{phase}.json",
        )
        summary = _teacher_run_summary(
            run_id=run.info.run_id,
            run_name_value=run_name_value,
            teacher_variant=teacher_variant,
            checkpoint_out=checkpoint_out,
            best_val_select=val_metrics,
            test_metrics=test_metrics,
            stm32sim=stm32sim,
            mcu_shortlist=shortlist_audit,
            quantize_dynamic=quantize_dynamic_flag,
            extra={
                "phase": phase,
                "inference_seconds_10s": latency_seconds,
            },
        )
        if teacher_ref_result is not None:
            summary.update(_teacher_metric_delta(teacher_ref_result, summary))
        if proxy_metrics:
            summary["proxy8k_val_select_pesq"] = proxy_metrics.get("val_select", {}).get("pesq_mean")
            summary["proxy8k_val_select_stoi"] = proxy_metrics.get("val_select", {}).get("stoi_mean")
            summary["proxy8k_val_select_sisdr"] = proxy_metrics.get("val_select", {}).get("sisdr_mean")
            summary["proxy8k_test_pesq"] = proxy_metrics.get("test", {}).get("pesq_mean")
            summary["proxy8k_test_stoi"] = proxy_metrics.get("test", {}).get("stoi_mean")
            summary["proxy8k_test_sisdr"] = proxy_metrics.get("test", {}).get("sisdr_mean")
        if extra_metrics:
            summary.update(extra_metrics)
        return summary
    except KeyboardInterrupt:
        run_status = "KILLED"
        raise
    except BaseException:
        run_status = "FAILED"
        raise
    finally:
        mlflow.end_run(status=run_status)


def run_teacher16k_fp32_ref_phase(
    args: argparse.Namespace,
    *,
    val_select_csv: str,
    test_csv: str,
) -> dict[str, Any]:
    existing = find_finished_run(
        args.mlflow_uri,
        args.teacher_audit_experiment_name,
        "metricgan_plus-16k-fp32-exact-ref",
        phase="teacher16k_fp32_ref",
    )
    if existing and args.resume:
        return teacher_audit_result_from_existing(existing)

    eval_device = args.device
    model = MetricGANPlusAdapter("small").to(eval_device)
    proxy_model = ResampledTeacherWrapper(
        model,
        input_sample_rate=8000,
        model_sample_rate=16000,
        output_sample_rate=8000,
        output_device=eval_device,
    )
    proxy_metrics = {
        "val_select": evaluate_manifest(
            proxy_model,
            val_select_csv,
            eval_device,
            sample_rate=8000,
            compute_dnsmos=False,
            compute_composite=False,
            batch_size=4,
            cache_audio=True,
            progress_callback=lambda message: campaign_log(f"teacher16k_fp32_ref[8kproxy]: {message}"),
        ),
        "test": evaluate_manifest(
            proxy_model,
            test_csv,
            eval_device,
            sample_rate=8000,
            compute_dnsmos=False,
            compute_composite=False,
            batch_size=4,
            cache_audio=True,
            progress_callback=lambda message: campaign_log(f"teacher16k_fp32_ref[8kproxy]: {message}"),
        ),
    }
    return _run_teacher_eval(
        args=args,
        phase="teacher16k_fp32_ref",
        run_name_value="metricgan_plus-16k-fp32-exact-ref",
        model=model,
        eval_device=eval_device,
        eval_sample_rate=16000,
        val_select_csv=val_select_csv,
        test_csv=test_csv,
        teacher_variant="metricgan_plus_16k_fp32_exact",
        quantize_dynamic_flag=False,
        stm32_weight_bits=32,
        proxy_metrics=proxy_metrics,
        extra_metrics={
            "teacher_accuracy_drop_pesq": 0.0,
            "teacher_accuracy_drop_stoi": 0.0,
            "teacher_accuracy_drop_sisdr": 0.0,
        },
    )


def run_teacher16k_portable_proxy_fp32_phase(
    args: argparse.Namespace,
    *,
    val_select_csv: str,
    test_csv: str,
    teacher16k_ref: dict[str, Any],
) -> dict[str, Any]:
    model = build_metricgan_standalone(
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        win_length=320,
        variant="small",
        native8k=False,
        init_from_pretrained=True,
    )
    return _run_teacher_eval(
        args=args,
        phase="teacher16k_int8_bench",
        run_name_value="metricgan_plus-16k-fp32-portable-proxy",
        model=model,
        eval_device="cpu",
        eval_sample_rate=16000,
        val_select_csv=val_select_csv,
        test_csv=test_csv,
        teacher_variant="metricgan_plus_16k_fp32_portable_proxy",
        quantize_dynamic_flag=False,
        stm32_weight_bits=32,
        teacher_ref_result=teacher16k_ref,
    )


def run_teacher16k_int8_bench_phase(
    args: argparse.Namespace,
    *,
    val_select_csv: str,
    test_csv: str,
    teacher16k_ref: dict[str, Any],
) -> dict[str, Any]:
    portable_proxy_fp32 = run_teacher16k_portable_proxy_fp32_phase(
        args,
        val_select_csv=val_select_csv,
        test_csv=test_csv,
        teacher16k_ref=teacher16k_ref,
    )
    model = build_metricgan_standalone(
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        win_length=320,
        variant="small",
        native8k=False,
        init_from_pretrained=True,
    )
    quantized_model = dynamic_quantize_metricgan(model)
    summary = _run_teacher_eval(
        args=args,
        phase="teacher16k_int8_bench",
        run_name_value="metricgan_plus-16k-int8-portable-proxy",
        model=quantized_model,
        eval_device="cpu",
        eval_sample_rate=16000,
        val_select_csv=val_select_csv,
        test_csv=test_csv,
        teacher_variant="metricgan_plus_16k_int8_portable_proxy",
        quantize_dynamic_flag=True,
        stm32_weight_bits=8,
        sim_model=model,
        teacher_ref_result=teacher16k_ref,
    )
    quantization_drop = _teacher_metric_delta(portable_proxy_fp32, summary, prefix="quantization_drop")
    summary.update(quantization_drop)
    summary["portable_proxy_fp32"] = portable_proxy_fp32
    if summary.get("run_id"):
        client = MlflowClient(tracking_uri=args.mlflow_uri)
        for key, value in quantization_drop.items():
            client.log_metric(summary["run_id"], key, value)
        client.set_tag(summary["run_id"], "audit_only", "true")
    return summary


def teacher8k_native_specs(
    args: argparse.Namespace,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    test_csv: str,
) -> list[ExperimentConfig]:
    teacher_ref = _search_best_teacher_run(args)
    return [
        ExperimentConfig(
            train_csv=train_fit_csv,
            val_rank_csv=val_rank_csv,
            val_select_csv=val_select_csv,
            test_csv=test_csv,
            checkpoint_out="checkpoints/teacher_audit/metricgan_plus_native8k_small.pt",
            model_family="metricgan_plus_native8k",
            variant="small",
            loss_recipe="R1",
            run_name="metricgan_plus_native8k-small-fp32",
            phase="teacher8k_native_train",
            epochs=args.epochs_teacher8k_train,
            lr=5e-4,
            segment_len=16000,
            seed=0,
            scheduler="plateau",
            lr_patience=args.teacher8k_lr_patience,
            early_stop_patience=args.teacher8k_early_stop_patience,
            min_epochs=args.teacher8k_min_epochs,
            eval_every=2,
            device=args.device,
            mlflow_uri=args.mlflow_uri,
            mlflow_artifact_root=args.mlflow_artifact_root,
            experiment_name=args.teacher_audit_experiment_name,
            teacher_source_run_id=teacher_ref["run_id"] if teacher_ref else None,
            teacher_variant="metricgan_plus_native8k_fp32",
            audit_only=True,
            eval_dnsmos=False,
            rank_compute_composite=False,
            select_compute_composite=False,
            mcu_profile=args.stm32_profile,
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            win_length=160,
        )
    ]


def run_teacher8k_native_train_phase(
    args: argparse.Namespace,
    *,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    test_csv: str,
    teacher16k_ref: dict[str, Any],
) -> dict[str, Any]:
    results = run_phase(
        "teacher8k_native_train",
        teacher8k_native_specs(args, train_fit_csv, val_rank_csv, val_select_csv, test_csv),
        args,
        experiment_name=args.teacher_audit_experiment_name,
    )
    results = attach_mcu_shortlist_audits(args, results)
    proxy_reference = _teacher_proxy_from_ref(teacher16k_ref)
    for result in results:
        deltas = _teacher_metric_delta(proxy_reference, result)
        result.update(deltas)
        client = MlflowClient(tracking_uri=args.mlflow_uri)
        for key, value in deltas.items():
            client.log_metric(result["run_id"], key, value)
        client.set_tag(result["run_id"], "audit_only", "true")
    return results[0]


def run_teacher8k_native_int8_bench_phase(
    args: argparse.Namespace,
    *,
    val_select_csv: str,
    test_csv: str,
    teacher16k_ref: dict[str, Any],
    teacher8k_fp32: dict[str, Any],
) -> dict[str, Any]:
    model, _ = load_model_from_checkpoint(
        teacher8k_fp32["checkpoint_out"],
        device="cpu",
        model_family="metricgan_plus_native8k",
        variant="small",
    )
    fp32_reference = dict(teacher8k_fp32)
    quantized = dynamic_quantize_metricgan(model)
    summary = _run_teacher_eval(
        args=args,
        phase="teacher8k_native_int8_bench",
        run_name_value="metricgan_plus_native8k-small-int8",
        model=quantized,
        eval_device="cpu",
        eval_sample_rate=8000,
        val_select_csv=val_select_csv,
        test_csv=test_csv,
        teacher_variant="metricgan_plus_native8k_int8",
        quantize_dynamic_flag=True,
        stm32_weight_bits=8,
        sim_model=model,
        checkpoint_out=teacher8k_fp32["checkpoint_out"],
        teacher_ref_result=_teacher_proxy_from_ref(teacher16k_ref),
        extra_metrics={},
    )
    quantization_drop = _teacher_metric_delta(fp32_reference, summary, prefix="quantization_drop")
    summary.update(quantization_drop)
    if summary.get("run_id"):
        client = MlflowClient(tracking_uri=args.mlflow_uri)
        for key, value in quantization_drop.items():
            client.log_metric(summary["run_id"], key, value)
        client.set_tag(summary["run_id"], "audit_only", "true")
    return summary


def choose_teacher_mcu_decision(
    teacher16k_ref: dict[str, Any],
    teacher16k_int8: dict[str, Any],
    teacher8k_fp32: dict[str, Any],
    teacher8k_int8: dict[str, Any] | None,
    *,
    native8k_drop_max: float = 0.15,
    native8k_quant_drop_max: float = 0.05,
) -> dict[str, Any]:
    native8k_drop = float(teacher8k_fp32.get("teacher_accuracy_drop_pesq") or float("inf"))
    if int((teacher16k_int8.get("mcu_shortlist") or {}).get("supported_profile_count") or 0) == 0:
        direct_reason = "teacher16k_portable_int8_has_no_supported_profiles"
    else:
        direct_reason = "teacher16k_portable_int8_has_supported_profiles"
    candidates: list[dict[str, Any]] = []
    if native8k_drop <= native8k_drop_max:
        candidates.append(_with_stm32_recommendation(teacher8k_fp32))
    if teacher8k_int8 is not None:
        quant_drop = float(teacher8k_int8.get("quantization_drop_pesq") or float("inf"))
        if quant_drop <= native8k_quant_drop_max:
            candidates.append(_with_stm32_recommendation(teacher8k_int8))
    viable = [candidate for candidate in candidates if _stm32_sim_is_eligible(candidate)]
    if not viable:
        return {
            "next_action": "direct_teacher_not_viable",
            "reason": direct_reason if not candidates else "no_teacher_variant_passed_accuracy_and_mcu_gates",
            "teacher16k_fp32": teacher16k_ref,
            "teacher16k_int8": teacher16k_int8,
            "teacher8k_fp32": teacher8k_fp32,
            "teacher8k_int8": teacher8k_int8,
        }
    ranked = sorted(viable, key=_teacher_candidate_sort_key, reverse=True)
    return {
        "next_action": "teacher_variant_best_audit",
        "reason": "teacher_variant_passed_accuracy_and_mcu_gates",
        "winner": ranked[0],
        "ranked_variants": ranked,
        "teacher16k_fp32": teacher16k_ref,
        "teacher16k_int8": teacher16k_int8,
        "teacher8k_fp32": teacher8k_fp32,
        "teacher8k_int8": teacher8k_int8,
    }


def stm32_result_from_existing(existing: dict[str, Any]) -> dict[str, Any]:
    summary = summary_from_existing(existing)
    metrics = existing.get("metrics", {})
    params = existing.get("params", {})
    summary.update(
        {
            "guidance_classic": params.get("guidance_classic"),
            "qat": params.get("qat"),
            "mcu_profile": params.get("mcu_profile"),
            "teacher_source_run_id": params.get("teacher_source_run_id"),
            "classic_gap_vs_spectral_gating": metrics.get("classic_gap_vs_spectral_gating"),
            "teacher_gap_pesq": metrics.get("teacher_gap_pesq"),
        }
    )
    shortlist = _mcu_shortlist_summary_from_raw(metrics, existing.get("tags", {}))
    if shortlist is not None:
        summary["mcu_shortlist"] = shortlist
    return summary


def annotate_stm32_gap_metrics(
    tracking_uri: str,
    results: list[dict[str, Any]],
    *,
    classic_pesq: float,
    teacher_pesq: float | None,
) -> None:
    client = MlflowClient(tracking_uri=tracking_uri)
    for result in results:
        metric_value = result.get("best_val_select_pesq")
        if metric_value is None and result.get("test_metrics"):
            metric_value = result["test_metrics"].get("pesq_mean")
        if metric_value is None or not result.get("run_id"):
            continue
        client.log_metric(result["run_id"], "classic_gap_vs_spectral_gating", float(metric_value) - classic_pesq)
        if teacher_pesq is not None:
            client.log_metric(result["run_id"], "teacher_gap_pesq", teacher_pesq - float(metric_value))


def run_or_resume(config: ExperimentConfig, *, resume: bool) -> dict[str, Any]:
    config.run_name = config.run_name or run_name(config)
    existing = None
    if resume:
        existing = find_finished_run(
            tracking_uri=config.mlflow_uri,
            experiment_name=config.experiment_name,
            run_name=config.run_name,
            phase=config.phase,
        )
    if existing:
        result = summary_from_existing(existing)
    else:
        failed_attempts = count_runs_by_status(
            tracking_uri=config.mlflow_uri,
            experiment_name=config.experiment_name,
            run_name=config.run_name,
            phase=config.phase,
            statuses=("FAILED",),
        )
        if failed_attempts >= 3:
            raise RuntimeError(
                f"Run {config.run_name} failed {failed_attempts} times. Manual intervention required before retrying."
            )
        result = run_experiment(config)

    result.update(
        {
            "model_family": config.model_family,
            "variant": config.variant,
            "loss_recipe": config.loss_recipe,
            "seed": config.seed,
            "lr": config.lr,
            "segment_len": config.segment_len,
            "scheduler": config.scheduler,
            "phase": config.phase,
            "run_name": config.run_name,
            "checkpoint_out": config.checkpoint_out,
        }
    )
    return result


def find_finished_run_across_experiments(
    tracking_uri: str,
    experiment_names: tuple[str, ...],
    *,
    run_name: str,
    phase: str | None = None,
) -> dict[str, Any] | None:
    client = MlflowClient(tracking_uri=tracking_uri)
    matched_runs: list[Any] = []
    filter_parts = [f"attributes.run_name = '{run_name}'", "attributes.status = 'FINISHED'"]
    if phase:
        filter_parts.append(f"tags.phase = '{phase}'")
    filter_string = " and ".join(filter_parts)
    for experiment_name in experiment_names:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            continue
        matched_runs.extend(
            client.search_runs(
                [experiment.experiment_id],
                filter_string=filter_string,
                max_results=20,
                order_by=["attributes.start_time DESC"],
            )
        )
    for run in sorted(matched_runs, key=lambda item: item.info.start_time or 0, reverse=True):
        if run.data.tags.get("audit.invalidated") == "true":
            continue
        return {
            "run_id": run.info.run_id,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
        }
    return None


def search_finished_results(
    tracking_uri: str,
    experiment_name: str,
    *,
    phase: str,
) -> list[dict[str, Any]]:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.status = 'FINISHED' and tags.phase = '{phase}'",
        order_by=["attributes.start_time DESC"],
        max_results=500,
    )
    results: list[dict[str, Any]] = []
    for run in runs:
        if run.data.tags.get("audit.invalidated") == "true":
            continue
        summary = summary_from_existing(
            {
                "run_id": run.info.run_id,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "tags": dict(run.data.tags),
            }
        )
        summary["checkpoint_out"] = run.data.params.get("checkpoint_out")
        summary["lr"] = float(run.data.params["lr"]) if "lr" in run.data.params else None
        summary["segment_len"] = int(run.data.params["segment_len"]) if "segment_len" in run.data.params else None
        summary["scheduler"] = run.data.params.get("scheduler")
        summary["phase"] = run.data.tags.get("phase")
        summary["source_run_id"] = run.data.params.get("source_run_id")
        summary["source_run_name"] = run.data.params.get("source_run_name")
        shortlist = _mcu_shortlist_summary_from_raw(run.data.metrics, run.data.tags)
        if shortlist is not None:
            summary["mcu_shortlist"] = shortlist
        results.append(summary)
    return results


def search_finished_results_across_experiments(
    tracking_uri: str,
    experiment_names: tuple[str, ...],
    *,
    phase: str,
) -> list[dict[str, Any]]:
    seen_run_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    for experiment_name in experiment_names:
        for result in search_finished_results(tracking_uri, experiment_name, phase=phase):
            if result["run_id"] in seen_run_ids:
                continue
            seen_run_ids.add(result["run_id"])
            results.append(result)
    return results


def pick_top_raw_by_family(
    tracking_uri: str,
    experiment_name: str,
    *,
    phases: tuple[str, ...] = ("phase0", "phase1"),
    count_per_family: int = 2,
) -> list[dict[str, Any]]:
    raw_results: list[dict[str, Any]] = []
    for phase in phases:
        raw_results.extend(search_finished_results(tracking_uri, experiment_name, phase=phase))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in raw_results:
        if not result.get("model_family"):
            continue
        if result.get("postfilter_mode") not in (None, "none"):
            continue
        grouped[str(result["model_family"])].append(result)

    chosen: list[dict[str, Any]] = []
    for family, group in grouped.items():
        unique: dict[tuple[Any, ...], dict[str, Any]] = {}
        for result in sorted(group, key=score_value, reverse=True):
            key = (
                result["model_family"],
                result["variant"],
                result["lr"],
                result["segment_len"],
                result["loss_recipe"],
                result.get("scheduler"),
            )
            if key not in unique:
                unique[key] = result
            if len(unique) >= count_per_family:
                break
        chosen.extend(unique.values())
    return chosen


def gating_stage1_specs(args: argparse.Namespace, val_select_csv: str) -> list[PostfilterEvalSpec]:
    specs: list[PostfilterEvalSpec] = []
    top_raw = pick_top_raw_by_family(
        args.mlflow_uri,
        args.experiment_name,
        phases=("phase0", "phase1"),
        count_per_family=2,
    )
    campaign_log(
        "gating_stage1 selected "
        f"{len(top_raw)} raw source checkpoints from experiment {args.experiment_name}"
    )
    variants = [
        ("none", "medium"),
        ("sg_residual_soft", "light"),
        ("sg_residual_soft", "medium"),
        ("sg_input_floor", "light"),
        ("sg_input_floor", "medium"),
    ]
    for result in top_raw:
        checkpoint_out = result.get("checkpoint_out")
        if not checkpoint_out:
            campaign_log(f"skipping source run without checkpoint_out: {result.get('run_name')}")
            continue
        campaign_log(
            f"source {result.get('run_name')} family={result.get('model_family')} "
            f"variant={result.get('variant')} checkpoint={checkpoint_out}"
        )
        for postfilter_mode, postfilter_preset in variants:
            specs.append(
                PostfilterEvalSpec(
                    source_run_id=str(result["run_id"]),
                    source_run_name=str(result["run_name"]),
                    checkpoint_out=str(checkpoint_out),
                    model_family=str(result["model_family"]),
                    variant=str(result["variant"]),
                    loss_recipe=str(result["loss_recipe"]),
                    scheduler=result.get("scheduler"),
                    lr=float(result["lr"]),
                    segment_len=int(result["segment_len"]),
                    seed=result.get("seed"),
                    phase="gating_stage1",
                    val_select_csv=val_select_csv,
                    mlflow_uri=args.mlflow_uri,
                    mlflow_artifact_root=args.mlflow_artifact_root,
                    experiment_name=args.gating_experiment_name,
                    device=args.device,
                    benchmark_repeats=1,
                    postfilter_mode=postfilter_mode,
                    postfilter_preset=postfilter_preset,
                )
            )
    return specs


def top_families(results: list[dict[str, Any]], count: int = 2) -> list[str]:
    best_by_family: dict[str, float] = {}
    for result in results:
        family = result["model_family"]
        best_by_family[family] = max(best_by_family.get(family, float("-inf")), score_value(result))
    return [family for family, _ in sorted(best_by_family.items(), key=lambda item: item[1], reverse=True)[:count]]


def best_segment_for_family(results: list[dict[str, Any]], family: str) -> int:
    subset = [result for result in results if result["model_family"] == family]
    subset.sort(key=score_value, reverse=True)
    return int(subset[0]["segment_len"])


def config_key_without_seed(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        result["model_family"],
        result["variant"],
        result["lr"],
        result["segment_len"],
        result["loss_recipe"],
        result["scheduler"],
    )


def pick_top_unique_configs(results: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    chosen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for result in sorted(results, key=score_value, reverse=True):
        key = config_key_without_seed(result)
        if key not in chosen:
            chosen[key] = result
        if len(chosen) >= count:
            break
    return list(chosen.values())


def aggregate_phase3(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[config_key_without_seed(result)].append(result)

    aggregates: list[dict[str, Any]] = []
    for key, group in grouped.items():
        pesq_values = [float(item["best_val_select_pesq"]) for item in group if item.get("best_val_select_pesq") is not None]
        dnsmos_values = [
            float(item["best_val_select_dnsmos_ovr"])
            for item in group
            if item.get("best_val_select_dnsmos_ovr") is not None
        ]
        latency_values = [
            float(item["inference_seconds_10s"])
            for item in group
            if item.get("inference_seconds_10s") is not None
        ]
        pesq = mean(pesq_values)
        dnsmos = mean(dnsmos_values) if dnsmos_values else float("-inf")
        latency = mean(latency_values) if latency_values else float("inf")
        aggregates.append(
            {
                "key": key,
                "mean_pesq": pesq,
                "mean_dnsmos_ovr": dnsmos,
                "mean_latency_10s": latency,
                "prototype": group[0],
            }
        )
    aggregates.sort(key=lambda item: item["mean_pesq"], reverse=True)
    return aggregates


def select_phase3_winner(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    winner = aggregates[0]
    if len(aggregates) == 1:
        return winner

    runner_up = aggregates[1]
    if abs(winner["mean_pesq"] - runner_up["mean_pesq"]) < 0.02:
        if runner_up["mean_dnsmos_ovr"] > winner["mean_dnsmos_ovr"]:
            winner = runner_up
        elif runner_up["mean_dnsmos_ovr"] == winner["mean_dnsmos_ovr"] and runner_up["mean_latency_10s"] < winner["mean_latency_10s"]:
            winner = runner_up
    return winner


def maybe_limit(specs: list[ExperimentConfig], max_runs: int | None) -> list[ExperimentConfig]:
    return specs if max_runs is None else specs[:max_runs]


def phase0_specs(args: argparse.Namespace, train_fit_csv: str, val_rank_csv: str, val_select_csv: str) -> list[ExperimentConfig]:
    specs: list[ExperimentConfig] = []
    for family in ("atennuate", "fullsubnet_plus", "mp_senet", "cmgan_small"):
        for segment_len in (16000, 32000):
            specs.append(
                ExperimentConfig(
                    train_csv=train_fit_csv,
                    val_rank_csv=val_rank_csv,
                    val_select_csv=val_select_csv,
                    checkpoint_out=f"checkpoints/{family}_smoke_seg{segment_len}.pt",
                    model_family=family,
                    variant="small",
                    loss_recipe="R1",
                    phase="phase0",
                    epochs=args.epochs_smoke,
                    segment_len=segment_len,
                    seed=0,
                    eval_every=1,
                    min_epochs=1,
                    early_stop_patience=0,
                    device=args.device,
                    mlflow_uri=args.mlflow_uri,
                    mlflow_artifact_root=args.mlflow_artifact_root,
                    experiment_name=args.experiment_name,
                )
            )
    return specs


def phase1_specs(args: argparse.Namespace, train_fit_csv: str, val_rank_csv: str, val_select_csv: str) -> list[ExperimentConfig]:
    specs: list[ExperimentConfig] = []
    for family in ("atennuate", "fullsubnet_plus", "mp_senet", "cmgan_small"):
        for variant in ("small", "base"):
            for lr in (1e-3, 5e-4, 2e-4):
                for segment_len in (16000, 32000):
                    for seed in (0, 1):
                        specs.append(
                            ExperimentConfig(
                                train_csv=train_fit_csv,
                                val_rank_csv=val_rank_csv,
                                val_select_csv=val_select_csv,
                                checkpoint_out=f"checkpoints/{family}_{variant}_lr{lr:g}_seg{segment_len}_s{seed}.pt",
                                model_family=family,
                                variant=variant,
                                loss_recipe="R1",
                                phase="phase1",
                                epochs=args.epochs_phase1,
                                lr=lr,
                                segment_len=segment_len,
                                seed=seed,
                                benchmark_repeats=1,
                                eval_dnsmos=False,
                                rank_compute_composite=False,
                                select_compute_composite=False,
                                device=args.device,
                                mlflow_uri=args.mlflow_uri,
                                mlflow_artifact_root=args.mlflow_artifact_root,
                                experiment_name=args.experiment_name,
                            )
                        )
    return specs


def phase2_specs(
    args: argparse.Namespace,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    phase1_results: list[dict[str, Any]],
) -> list[ExperimentConfig]:
    families = top_families(phase1_results, count=2)
    specs: list[ExperimentConfig] = []
    for family in families:
        segment_len = best_segment_for_family(phase1_results, family)
        for variant in ("small", "base"):
            for loss_recipe in ("R1", "R2", "R3", "R4"):
                for scheduler in ("plateau", "cosine"):
                    for seed in (0, 1):
                        lr = 5e-4 if family != "cmgan_small" else 2e-4
                        specs.append(
                            ExperimentConfig(
                                train_csv=train_fit_csv,
                                val_rank_csv=val_rank_csv,
                                val_select_csv=val_select_csv,
                                checkpoint_out=f"checkpoints/{family}_{variant}_{loss_recipe}_{scheduler}_s{seed}.pt",
                                model_family=family,
                                variant=variant,
                                loss_recipe=loss_recipe,
                                scheduler=scheduler,
                                phase="phase2",
                                epochs=args.epochs_phase2,
                                lr=lr,
                                segment_len=segment_len,
                                seed=seed,
                                device=args.device,
                                mlflow_uri=args.mlflow_uri,
                                mlflow_artifact_root=args.mlflow_artifact_root,
                                experiment_name=args.experiment_name,
                            )
                        )
    return specs


def phase3_specs(
    args: argparse.Namespace,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    phase2_results: list[dict[str, Any]],
) -> list[ExperimentConfig]:
    specs: list[ExperimentConfig] = []
    for base in pick_top_unique_configs(phase2_results, count=8):
        family = base["model_family"]
        for seed in (2, 3, 4):
            specs.append(
                ExperimentConfig(
                    train_csv=train_fit_csv,
                    val_rank_csv=val_rank_csv,
                    val_select_csv=val_select_csv,
                    checkpoint_out=f"checkpoints/{family}_{base['variant']}_{base['loss_recipe']}_{base['scheduler']}_robust_s{seed}.pt",
                    model_family=family,
                    variant=base["variant"],
                    loss_recipe=base["loss_recipe"],
                    scheduler=base["scheduler"],
                    phase="phase3",
                    epochs=args.epochs_phase3,
                    lr=float(base["lr"]),
                    segment_len=int(base["segment_len"]),
                    seed=seed,
                    device=args.device,
                    mlflow_uri=args.mlflow_uri,
                    mlflow_artifact_root=args.mlflow_artifact_root,
                    experiment_name=args.experiment_name,
                )
            )
    return specs


def phase4_specs(args: argparse.Namespace, winner: dict[str, Any], train_csv: str, test_csv: str) -> list[ExperimentConfig]:
    prototype = winner["prototype"]
    family = prototype["model_family"]
    specs: list[ExperimentConfig] = []
    for seed in range(10, 18):
        specs.append(
            ExperimentConfig(
                train_csv=train_csv,
                test_csv=test_csv,
                checkpoint_out=f"checkpoints/{family}_{prototype['variant']}_{prototype['loss_recipe']}_{prototype['scheduler']}_final_s{seed}.pt",
                model_family=family,
                variant=prototype["variant"],
                loss_recipe=prototype["loss_recipe"],
                scheduler=prototype["scheduler"],
                phase="phase4",
                epochs=args.epochs_phase4,
                lr=float(prototype["lr"]),
                segment_len=int(prototype["segment_len"]),
                seed=seed,
                device=args.device,
                mlflow_uri=args.mlflow_uri,
                mlflow_artifact_root=args.mlflow_artifact_root,
                experiment_name=args.experiment_name,
                early_stop_patience=0,
                min_epochs=args.epochs_phase4,
                eval_every=args.epochs_phase4,
                log_torch_model=True,
            )
        )
    return specs


def stage1_summary_from_existing(existing: dict[str, Any]) -> dict[str, Any]:
    metrics = existing.get("metrics", {})
    params = existing.get("params", {})
    return {
        "run_id": existing["run_id"],
        "run_name": existing["tags"].get("mlflow.runName"),
        "model_family": params.get("model_family"),
        "variant": params.get("variant"),
        "loss_recipe": params.get("loss_recipe"),
        "scheduler": params.get("scheduler"),
        "seed": int(params["seed"]) if "seed" in params else None,
        "lr": float(params["lr"]) if "lr" in params else None,
        "segment_len": int(params["segment_len"]) if "segment_len" in params else None,
        "postfilter_mode": params.get("postfilter_mode"),
        "postfilter_preset": params.get("postfilter_preset"),
        "train_postfilter": params.get("train_postfilter"),
        "spectral_native_gate": params.get("spectral_native_gate"),
        "source_run_id": params.get("source_run_id"),
        "source_run_name": params.get("source_run_name"),
        "best_val_select_pesq": metrics.get("best/val_select_pesq_mean"),
        "best_val_select_stoi": metrics.get("best/val_select_stoi_mean"),
        "best_val_select_dnsmos_ovr": metrics.get("best/val_select_dnsmos_ovr_mean"),
        "inference_seconds_10s": metrics.get("best/inference_seconds_10s"),
    }


def run_postfilter_eval(spec: PostfilterEvalSpec, *, resume: bool) -> dict[str, Any]:
    def progress(message: str) -> None:
        campaign_log(f"{spec.phase}: {spec.run_name}: {message}")

    campaign_log(
        f"{spec.phase}: preparing {spec.run_name} "
        f"(source={spec.source_run_name}, pf={spec.postfilter_mode}/{spec.postfilter_preset})"
    )
    existing = None
    if resume:
        existing = find_finished_run_across_experiments(
            tracking_uri=spec.mlflow_uri,
            experiment_names=(spec.experiment_name, "Default"),
            run_name=spec.run_name,
            phase=spec.phase,
        )
    if existing:
        campaign_log(f"{spec.phase}: reusing finished run for {spec.run_name}")
        result = stage1_summary_from_existing(existing)
        result.update(
            {
                "source_run_id": spec.source_run_id,
                "source_run_name": spec.source_run_name,
                "checkpoint_out": spec.checkpoint_out,
            }
        )
        return result

    checkpoint_path = Path(spec.checkpoint_out)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for stage1 gating eval: {checkpoint_path}")
    campaign_log(f"{spec.phase}: checkpoint present for {spec.run_name}: {checkpoint_path}")

    experiment_id = configure_mlflow(spec.mlflow_uri, spec.experiment_name, spec.mlflow_artifact_root)
    run = mlflow.start_run(
        run_name=spec.run_name,
        experiment_id=experiment_id,
        nested=True,
        tags={
            "phase": spec.phase,
            "run_type": "child",
            "source_run_id": spec.source_run_id,
            "source_run_name": spec.source_run_name,
        },
    )
    campaign_log(f"{spec.phase}: started child run {run.info.run_id} for {spec.run_name}")
    sample_rows = read_pair_manifest(spec.val_select_csv)
    sample_path = sample_rows[0].noisy
    campaign_log(f"{spec.phase}: loaded manifest rows={len(sample_rows)} for {spec.run_name}")
    sample_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_{spec.run_name}_samples"
    if sample_dir.exists():
        shutil.rmtree(sample_dir, ignore_errors=True)
    run_status = "FINISHED"
    try:
        campaign_log(f"{spec.phase}: loading checkpoint/model for {spec.run_name}")
        model, _ = load_model_from_checkpoint(
            checkpoint_path,
            device=spec.device,
            model_family=spec.model_family,
            variant=spec.variant,
            postfilter_mode=spec.postfilter_mode,
            postfilter_preset=spec.postfilter_preset,
            train_postfilter=spec.train_postfilter,
            spectral_native_gate=spec.spectral_native_gate,
        )
        params = {
            "source_run_id": spec.source_run_id,
            "source_run_name": spec.source_run_name,
            "checkpoint_out": spec.checkpoint_out,
            "model_family": spec.model_family,
            "variant": spec.variant,
            "loss_recipe": spec.loss_recipe,
            "scheduler": spec.scheduler,
            "lr": spec.lr,
            "segment_len": spec.segment_len,
            "seed": spec.seed,
            "phase": spec.phase,
            "postfilter_mode": spec.postfilter_mode,
            "postfilter_preset": spec.postfilter_preset,
            "train_postfilter": spec.train_postfilter,
            "spectral_native_gate": spec.spectral_native_gate,
        }
        mlflow.log_params(params)
        campaign_log(f"{spec.phase}: evaluating val_select for {spec.run_name}")
        metrics = evaluate_manifest(
            model,
            spec.val_select_csv,
            spec.device,
            sample_rate=16000,
            compute_dnsmos=False,
            compute_composite=False,
            sample_dir=sample_dir,
            sample_count=spec.sample_count,
            batch_size=8,
            cache_audio=True,
            progress_callback=progress,
        )
        campaign_log(
            f"{spec.phase}: metrics ready for {spec.run_name} "
            f"(pesq={metrics.get('pesq_mean')}, stoi={metrics.get('stoi_mean')})"
        )
        latency_seconds = benchmark_inference(
            model,
            sample_path=sample_path,
            device=spec.device,
            sample_rate=16000,
            duration_seconds=spec.benchmark_seconds,
            repeats=spec.benchmark_repeats,
        )
        campaign_log(f"{spec.phase}: latency benchmark done for {spec.run_name}: {latency_seconds:.4f}s")
        metrics["benchmark_latency_10s"] = latency_seconds
        mlflow.log_metrics(
            {
                "best/val_select_pesq_mean": metrics["pesq_mean"],
                "best/val_select_stoi_mean": metrics["stoi_mean"],
                "best/val_select_sisdr_mean": metrics["sisdr_mean"],
                "best/val_select_delta_snr_mean": metrics["delta_snr_mean"],
                "best/inference_seconds_10s": latency_seconds,
            }
        )
        if "csig_mean" in metrics:
            mlflow.log_metrics(
                {
                    "best/val_select_csig_mean": metrics["csig_mean"],
                    "best/val_select_cbak_mean": metrics["cbak_mean"],
                    "best/val_select_covl_mean": metrics["covl_mean"],
                }
            )
        if "dnsmos_ovr_mean" in metrics:
            mlflow.log_metrics(
                {
                    "best/val_select_dnsmos_sig_mean": metrics["dnsmos_sig_mean"],
                    "best/val_select_dnsmos_bak_mean": metrics["dnsmos_bak_mean"],
                    "best/val_select_dnsmos_ovr_mean": metrics["dnsmos_ovr_mean"],
                }
            )
        if sample_dir.exists():
            mlflow.log_artifacts(sample_dir.as_posix(), artifact_path="samples")
            shutil.rmtree(sample_dir, ignore_errors=True)
        log_dict_artifact(metrics, "reports/best_val_select_metrics.json")
        campaign_log(f"{spec.phase}: finished child run {run.info.run_id} for {spec.run_name}")
        return {
            "run_id": run.info.run_id,
            "run_name": spec.run_name,
            "model_family": spec.model_family,
            "variant": spec.variant,
            "loss_recipe": spec.loss_recipe,
            "scheduler": spec.scheduler,
            "seed": spec.seed,
            "lr": spec.lr,
            "segment_len": spec.segment_len,
            "phase": spec.phase,
            "source_run_id": spec.source_run_id,
            "source_run_name": spec.source_run_name,
            "checkpoint_out": spec.checkpoint_out,
            "postfilter_mode": spec.postfilter_mode,
            "postfilter_preset": spec.postfilter_preset,
            "train_postfilter": spec.train_postfilter,
            "spectral_native_gate": spec.spectral_native_gate,
            "best_val_select_pesq": metrics["pesq_mean"],
            "best_val_select_stoi": metrics["stoi_mean"],
            "best_val_select_dnsmos_ovr": metrics.get("dnsmos_ovr_mean"),
            "inference_seconds_10s": latency_seconds,
        }
    except KeyboardInterrupt:
        run_status = "KILLED"
        campaign_log(f"{spec.phase}: interrupted {spec.run_name}")
        raise
    except BaseException:
        run_status = "FAILED"
        campaign_log(f"{spec.phase}: failed {spec.run_name}")
        raise
    finally:
        if sample_dir.exists():
            shutil.rmtree(sample_dir, ignore_errors=True)
        mlflow.end_run(status=run_status)


def run_postfilter_phase(phase: str, specs: list[PostfilterEvalSpec], args: argparse.Namespace) -> list[dict[str, Any]]:
    campaign_log(f"starting {phase} with {len(specs)} specs")
    experiment_id = configure_mlflow(args.mlflow_uri, args.gating_experiment_name, args.mlflow_artifact_root)
    terminate_matching_runs(
        tracking_uri=args.mlflow_uri,
        experiment_name=args.gating_experiment_name,
        run_name=phase,
        phase=phase,
        run_type="parent",
    )
    previous_handlers = install_termination_handlers()
    parent_run = mlflow.start_run(
        run_name=phase,
        experiment_id=experiment_id,
        tags={"phase": phase, "run_type": "parent"},
    )
    results: list[dict[str, Any]] = []
    run_status = "FINISHED"
    try:
        limited_specs = maybe_limit(specs, args.max_runs)
        campaign_log(f"{phase}: executing {len(limited_specs)} specs after max-runs filter")
        for index, spec in enumerate(limited_specs, start=1):
            campaign_log(f"{phase}: [{index}/{len(limited_specs)}] {spec.run_name}")
            results.append(run_postfilter_eval(spec, resume=args.resume))
        log_dict_artifact({"phase": phase, "results": results}, f"campaign/{phase}_results.json")
        campaign_log(f"completed {phase} with {len(results)} results")
        return results
    except KeyboardInterrupt:
        run_status = "KILLED"
        campaign_log(f"{phase}: interrupted")
        raise
    except BaseException:
        run_status = "FAILED"
        campaign_log(f"{phase}: failed")
        raise
    finally:
        restore_termination_handlers(previous_handlers)
        mlflow.end_run(status=run_status)


def cascade_stage1_specs(args: argparse.Namespace, val_select_csv: str) -> list[CascadeStage1Spec]:
    variants = [
        ("none", "medium"),
        ("sg_residual_soft", "light"),
        ("sg_residual_soft", "medium"),
        ("sg_input_floor", "light"),
        ("sg_input_floor", "medium"),
    ]
    specs: list[CascadeStage1Spec] = []
    for postfilter_mode, postfilter_preset in variants:
        specs.append(
            CascadeStage1Spec(
                model_family="metricgan_plus",
                variant="small",
                phase="cascade_stage1",
                val_select_csv=val_select_csv,
                mlflow_uri=args.mlflow_uri,
                mlflow_artifact_root=args.mlflow_artifact_root,
                experiment_name=args.cascade_experiment_name,
                device=args.device,
                postfilter_mode=postfilter_mode,
                postfilter_preset=postfilter_preset,
            )
        )
    return specs


def run_cascade_stage1_eval(spec: CascadeStage1Spec, *, resume: bool) -> dict[str, Any]:
    def progress(message: str) -> None:
        campaign_log(f"{spec.phase}: {spec.run_name}: {message}")

    existing = None
    if resume:
        existing = find_finished_run_across_experiments(
            tracking_uri=spec.mlflow_uri,
            experiment_names=(spec.experiment_name, "Default"),
            run_name=spec.run_name,
            phase=spec.phase,
        )
    if existing:
        campaign_log(f"{spec.phase}: reusing finished run for {spec.run_name}")
        return stage1_summary_from_existing(existing)

    experiment_id = configure_mlflow(spec.mlflow_uri, spec.experiment_name, spec.mlflow_artifact_root)
    run = mlflow.start_run(
        run_name=spec.run_name,
        experiment_id=experiment_id,
        nested=True,
        tags={"phase": spec.phase, "run_type": "child"},
    )
    run_status = "FINISHED"
    sample_dir = Path("checkpoints/cascade") / f"{spec.run_name}_samples"
    sample_rows = read_pair_manifest(spec.val_select_csv)
    sample_path = sample_rows[0].noisy
    try:
        model = build_enhancer(
            spec.model_family,
            spec.variant,
            postfilter_mode=spec.postfilter_mode,
            postfilter_preset=spec.postfilter_preset,
            train_postfilter=False,
        ).to(spec.device)
        params = {
            "model_family": spec.model_family,
            "variant": spec.variant,
            "phase": spec.phase,
            "pretrained_source": METRICGAN_PLUS_SOURCE,
            "postfilter_mode": spec.postfilter_mode,
            "postfilter_preset": spec.postfilter_preset,
            "train_postfilter": False,
            "spectral_native_gate": False,
        }
        mlflow.log_params(params)
        metrics = evaluate_manifest(
            model,
            spec.val_select_csv,
            spec.device,
            sample_rate=16000,
            compute_dnsmos=False,
            compute_composite=False,
            sample_dir=sample_dir,
            sample_count=spec.sample_count,
            batch_size=8,
            cache_audio=True,
            progress_callback=progress,
        )
        latency_seconds = benchmark_inference(
            model,
            sample_path=sample_path,
            device=spec.device,
            sample_rate=16000,
            duration_seconds=spec.benchmark_seconds,
            repeats=spec.benchmark_repeats,
        )
        mlflow.log_metrics(
            {
                "best/val_select_pesq_mean": metrics["pesq_mean"],
                "best/val_select_stoi_mean": metrics["stoi_mean"],
                "best/val_select_sisdr_mean": metrics["sisdr_mean"],
                "best/val_select_delta_snr_mean": metrics["delta_snr_mean"],
                "best/inference_seconds_10s": latency_seconds,
            }
        )
        if sample_dir.exists():
            mlflow.log_artifacts(sample_dir.as_posix(), artifact_path="samples")
        log_dict_artifact(metrics, "reports/best_val_select_metrics.json")
        return {
            "run_id": run.info.run_id,
            "run_name": spec.run_name,
            "model_family": spec.model_family,
            "variant": spec.variant,
            "loss_recipe": "pretrained",
            "scheduler": None,
            "seed": None,
            "lr": None,
            "segment_len": None,
            "phase": spec.phase,
            "source_run_id": None,
            "source_run_name": spec.run_name,
            "checkpoint_out": None,
            "postfilter_mode": spec.postfilter_mode,
            "postfilter_preset": spec.postfilter_preset,
            "train_postfilter": False,
            "spectral_native_gate": False,
            "best_val_select_pesq": metrics["pesq_mean"],
            "best_val_select_stoi": metrics["stoi_mean"],
            "best_val_select_dnsmos_ovr": metrics.get("dnsmos_ovr_mean"),
            "inference_seconds_10s": latency_seconds,
        }
    except KeyboardInterrupt:
        run_status = "KILLED"
        campaign_log(f"{spec.phase}: interrupted {spec.run_name}")
        raise
    except BaseException:
        run_status = "FAILED"
        campaign_log(f"{spec.phase}: failed {spec.run_name}")
        raise
    finally:
        if sample_dir.exists():
            shutil.rmtree(sample_dir, ignore_errors=True)
        mlflow.end_run(status=run_status)


def run_cascade_stage1_phase(phase: str, specs: list[CascadeStage1Spec], args: argparse.Namespace) -> list[dict[str, Any]]:
    campaign_log(f"starting {phase} with {len(specs)} specs")
    experiment_id = configure_mlflow(args.mlflow_uri, args.cascade_experiment_name, args.mlflow_artifact_root)
    terminate_matching_runs(
        tracking_uri=args.mlflow_uri,
        experiment_name=args.cascade_experiment_name,
        run_name=phase,
        phase=phase,
        run_type="parent",
    )
    previous_handlers = install_termination_handlers()
    parent_run = mlflow.start_run(
        run_name=phase,
        experiment_id=experiment_id,
        tags={"phase": phase, "run_type": "parent"},
    )
    results: list[dict[str, Any]] = []
    run_status = "FINISHED"
    try:
        limited_specs = maybe_limit(specs, args.max_runs)
        campaign_log(f"{phase}: executing {len(limited_specs)} specs after max-runs filter")
        for index, spec in enumerate(limited_specs, start=1):
            campaign_log(f"{phase}: [{index}/{len(limited_specs)}] {spec.run_name}")
            results.append(run_cascade_stage1_eval(spec, resume=args.resume))
        log_dict_artifact({"phase": phase, "results": results}, f"campaign/{phase}_results.json")
        campaign_log(f"completed {phase} with {len(results)} results")
        return results
    except KeyboardInterrupt:
        run_status = "KILLED"
        campaign_log(f"{phase}: interrupted")
        raise
    except BaseException:
        run_status = "FAILED"
        campaign_log(f"{phase}: failed")
        raise
    finally:
        restore_termination_handlers(previous_handlers)
        mlflow.end_run(status=run_status)


def promote_cascade_stage1(stage1_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw = next((item for item in stage1_results if item.get("postfilter_mode") == "none"), None)
    if raw is None or raw.get("best_val_select_pesq") is None or raw.get("best_val_select_stoi") is None:
        return None

    winner = dict(raw)
    for candidate in stage1_results:
        if candidate.get("postfilter_mode") == "none":
            continue
        if candidate.get("best_val_select_pesq") is None or candidate.get("best_val_select_stoi") is None:
            continue
        delta_pesq = float(candidate["best_val_select_pesq"]) - float(raw["best_val_select_pesq"])
        delta_stoi = float(candidate["best_val_select_stoi"]) - float(raw["best_val_select_stoi"])
        if delta_pesq < 0.02 or delta_stoi < -0.01:
            continue
        if float(candidate["best_val_select_pesq"]) > float(winner["best_val_select_pesq"]):
            winner = dict(candidate)
            winner["delta_pesq_vs_raw"] = delta_pesq
            winner["delta_stoi_vs_raw"] = delta_stoi
    return winner


def cascade_stage2_specs(
    args: argparse.Namespace,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    stage1_results: list[dict[str, Any]],
) -> list[ExperimentConfig]:
    promoted = promote_cascade_stage1(stage1_results)
    if promoted is None:
        return []

    specs: list[ExperimentConfig] = []
    # Start with the highest-signal, lowest-cost probe set. Expand only if this refiner
    # proves it can beat the pretrained raw baseline on val_select.
    for variant, loss_recipe in (("small", "R1"), ("small", "R5")):
        for seed in (0, 1):
            specs.append(
                ExperimentConfig(
                    train_csv=train_fit_csv,
                    val_rank_csv=val_rank_csv,
                    val_select_csv=val_select_csv,
                    checkpoint_out=(
                        f"checkpoints/cascade/metricgan_plus_refiner_{variant}_{loss_recipe}_"
                        f"pf{promoted['postfilter_mode']}_{promoted['postfilter_preset']}_s{seed}.pt"
                    ),
                    model_family="metricgan_plus_refiner",
                    variant=variant,
                    loss_recipe=loss_recipe,
                    scheduler="plateau",
                    phase="cascade_stage2",
                    epochs=args.epochs_cascade_train,
                    lr=5e-4,
                    segment_len=32000,
                    seed=seed,
                    device=args.device,
                    mlflow_uri=args.mlflow_uri,
                    mlflow_artifact_root=args.mlflow_artifact_root,
                    experiment_name=args.cascade_experiment_name,
                    early_stop_patience=args.cascade_early_stop_patience,
                    min_epochs=args.cascade_min_epochs,
                    lr_patience=4,
                    eval_dnsmos=False,
                    postfilter_mode=str(promoted["postfilter_mode"]),
                    postfilter_preset=str(promoted["postfilter_preset"]),
                    train_postfilter=False,
                )
            )
    return specs


def best_finished_cascade_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    finished = [result for result in results if result.get("best_val_select_pesq") is not None]
    if not finished:
        return None
    return max(finished, key=score_value)


def cascade_stage2_expand_specs(
    args: argparse.Namespace,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    stage1_results: list[dict[str, Any]],
) -> list[ExperimentConfig]:
    promoted = promote_cascade_stage1(stage1_results)
    if promoted is None:
        return []

    specs: list[ExperimentConfig] = []
    for variant, loss_recipe in (("small", "R2"), ("small", "R6"), ("base", "R1"), ("base", "R5")):
        for seed in (0, 1):
            specs.append(
                ExperimentConfig(
                    train_csv=train_fit_csv,
                    val_rank_csv=val_rank_csv,
                    val_select_csv=val_select_csv,
                    checkpoint_out=(
                        f"checkpoints/cascade/metricgan_plus_refiner_{variant}_{loss_recipe}_"
                        f"pf{promoted['postfilter_mode']}_{promoted['postfilter_preset']}_s{seed}.pt"
                    ),
                    model_family="metricgan_plus_refiner",
                    variant=variant,
                    loss_recipe=loss_recipe,
                    scheduler="plateau",
                    phase="cascade_expand",
                    epochs=args.epochs_cascade_train,
                    lr=5e-4,
                    segment_len=32000,
                    seed=seed,
                    device=args.device,
                    mlflow_uri=args.mlflow_uri,
                    mlflow_artifact_root=args.mlflow_artifact_root,
                    experiment_name=args.cascade_experiment_name,
                    early_stop_patience=args.cascade_early_stop_patience,
                    min_epochs=args.cascade_min_epochs,
                    lr_patience=4,
                    eval_dnsmos=False,
                    postfilter_mode=str(promoted["postfilter_mode"]),
                    postfilter_preset=str(promoted["postfilter_preset"]),
                    train_postfilter=False,
                )
            )
    return specs


def choose_cascade_followup(
    args: argparse.Namespace,
    stage1_results: list[dict[str, Any]],
    stage2_results: list[dict[str, Any]],
) -> dict[str, Any]:
    classic_summary = summarize_classic_baselines(args.classic_baselines_xlsx)
    raw_best = best_finished_cascade_result(stage1_results)
    stage2_best = best_finished_cascade_result(stage2_results)
    if raw_best is None:
        raise ValueError("Cascade follow-up requires at least one finished stage1 result.")

    improve_threshold = float(args.cascade_improve_threshold)
    winner = raw_best
    winner_source = "cascade_stage1"
    if stage2_best is not None:
        raw_pesq = float(raw_best["best_val_select_pesq"])
        stage2_pesq = float(stage2_best["best_val_select_pesq"])
        if stage2_pesq >= raw_pesq + improve_threshold:
            winner = stage2_best
            winner_source = str(stage2_best.get("phase") or "cascade_stage2")

    top_classic_pesq = classic_summary.get("top_pesq")
    next_action = "cascade_test"
    reasons: list[str] = []
    if winner_source != "cascade_stage1":
        reasons.append("stage2_beat_stage1_threshold")
    else:
        reasons.append("stage1_raw_remains_best")
    if top_classic_pesq is not None:
        if float(raw_best["best_val_select_pesq"]) >= float(top_classic_pesq):
            reasons.append("raw_beats_classic_baselines")
        else:
            reasons.append("raw_below_classic_baselines")
            next_action = "cascade_expand"
        if float(winner["best_val_select_pesq"]) >= float(top_classic_pesq):
            next_action = "cascade_test"
    return {
        "winner": winner,
        "winner_source": winner_source,
        "next_action": next_action,
        "classic_summary": classic_summary,
        "stage1_best_pesq": raw_best.get("best_val_select_pesq"),
        "stage2_best_pesq": stage2_best.get("best_val_select_pesq") if stage2_best else None,
        "reasons": reasons,
    }


def test_score_value(result: dict[str, Any]) -> float:
    test_metrics = result.get("test_metrics") or {}
    pesq = test_metrics.get("pesq_mean")
    return float(pesq) if pesq is not None else float("-inf")


def best_finished_test_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    finished = [result for result in results if (result.get("test_metrics") or {}).get("pesq_mean") is not None]
    if not finished:
        return None
    return max(finished, key=test_score_value)


def tested_source_run_names(results: list[dict[str, Any]]) -> set[str]:
    tested: set[str] = set()
    for result in results:
        source_run_name = result.get("source_run_name") or result.get("winner_source")
        if source_run_name:
            tested.add(str(source_run_name))
    return tested


def search_supported_raw_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    classic_index = classic_pesq_index(args.classic_baselines_xlsx)
    raw_results: list[dict[str, Any]] = []
    for phase in ("phase3", "phase2", "phase1", "phase0"):
        raw_results.extend(search_finished_results(args.mlflow_uri, args.experiment_name, phase=phase))

    unique: dict[str, dict[str, Any]] = {}
    for result in sorted(raw_results, key=score_value, reverse=True):
        run_name_value = str(result.get("run_name") or "")
        if not run_name_value:
            continue
        if result.get("postfilter_mode") not in (None, "none"):
            continue
        if not result.get("checkpoint_out"):
            continue
        if result.get("best_val_select_pesq") is None:
            continue
        if run_name_value not in unique:
            unique[run_name_value] = result

    def sort_key(result: dict[str, Any]) -> tuple[float, float]:
        family_name = normalize_baseline_name(str(result.get("model_family") or ""))
        return (
            float(classic_index.get(family_name, float("-inf"))),
            score_value(result),
        )

    return sorted(unique.values(), key=sort_key, reverse=True)


def choose_cascade_post_test_followup(
    args: argparse.Namespace,
    initial_decision: dict[str, Any],
    test_results: list[dict[str, Any]],
    raw_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    best_test = best_finished_test_result(test_results)
    target_pesq = float(args.target_pesq)
    classic_summary = summarize_classic_baselines(args.classic_baselines_xlsx)
    tested_sources = tested_source_run_names(test_results)
    candidate_lookup = {str(item.get("run_name") or ""): item for item in raw_candidates}
    candidate_lookup[str(initial_decision["winner"].get("run_name") or "")] = initial_decision["winner"]

    current_reference = float(initial_decision["winner"].get("best_val_select_pesq") or float("-inf"))
    if best_test is not None:
        best_test_pesq = float(best_test["test_metrics"]["pesq_mean"])
        if best_test_pesq >= target_pesq:
            return {
                "next_action": "stop",
                "reasons": ["target_met"],
                "best_test": best_test,
                "tested_source_run_names": sorted(tested_sources),
                "target_pesq": target_pesq,
                "classic_summary": classic_summary,
            }
        source_name = str(best_test.get("source_run_name") or best_test.get("winner_source") or "")
        if source_name and source_name in candidate_lookup:
            current_reference = float(candidate_lookup[source_name].get("best_val_select_pesq") or current_reference)

    max_val_gap = float(args.auto_next_max_val_gap)
    ranked_candidates: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        source_name = str(candidate.get("run_name") or "")
        if not source_name or source_name in tested_sources:
            continue
        candidate_val = candidate.get("best_val_select_pesq")
        if candidate_val is None:
            continue
        if current_reference != float("-inf") and float(candidate_val) < current_reference - max_val_gap:
            continue
        ranked_candidates.append(candidate)

    if ranked_candidates:
        next_candidate = ranked_candidates[0]
        return {
            "next_action": "cascade_test",
            "next_candidate": next_candidate,
            "reasons": ["target_unmet", "next_best_supported_raw_candidate"],
            "best_test": best_test,
            "tested_source_run_names": sorted(tested_sources),
            "target_pesq": target_pesq,
            "classic_summary": classic_summary,
        }

    return {
        "next_action": "stop",
        "reasons": ["target_unmet", "no_high_value_candidates_left"],
        "best_test": best_test,
        "tested_source_run_names": sorted(tested_sources),
        "target_pesq": target_pesq,
        "classic_summary": classic_summary,
    }


def run_cascade_auto_next_phase(
    args: argparse.Namespace,
    initial_decision: dict[str, Any],
) -> dict[str, Any]:
    auto_results: list[dict[str, Any]] = []
    test_results = search_finished_results(args.mlflow_uri, args.cascade_experiment_name, phase="cascade_test")
    if not test_results and initial_decision["next_action"] != "cascade_stop":
        campaign_log(f"cascade auto-next: no finished test result found, evaluating {initial_decision['winner']['run_name']}")
        auto_results.append(run_cascade_test_phase(args, initial_decision["winner"]))
        test_results = search_finished_results(args.mlflow_uri, args.cascade_experiment_name, phase="cascade_test")

    raw_candidates = search_supported_raw_candidates(args)
    post_test_decision = choose_cascade_post_test_followup(args, initial_decision, test_results, raw_candidates)
    campaign_log(
        "cascade post-test decision: "
        f"next_action={post_test_decision['next_action']} "
        f"best_test_pesq={(post_test_decision.get('best_test') or {}).get('test_metrics', {}).get('pesq_mean')}"
    )

    extra_tests_remaining = max(int(args.auto_next_max_extra_tests), 0)
    while post_test_decision["next_action"] == "cascade_test" and extra_tests_remaining > 0:
        candidate = post_test_decision["next_candidate"]
        campaign_log(f"cascade auto-next: testing additional candidate {candidate['run_name']}")
        auto_results.append(run_cascade_test_phase(args, candidate))
        extra_tests_remaining -= 1
        test_results = search_finished_results(args.mlflow_uri, args.cascade_experiment_name, phase="cascade_test")
        post_test_decision = choose_cascade_post_test_followup(args, initial_decision, test_results, raw_candidates)
        campaign_log(
            "cascade post-test decision updated: "
            f"next_action={post_test_decision['next_action']} "
            f"best_test_pesq={(post_test_decision.get('best_test') or {}).get('test_metrics', {}).get('pesq_mean')}"
        )

    if post_test_decision["next_action"] == "cascade_test" and extra_tests_remaining == 0:
        post_test_decision = dict(post_test_decision)
        post_test_decision["next_action"] = "stop"
        post_test_decision["reasons"] = list(post_test_decision.get("reasons", [])) + ["auto_next_budget_exhausted"]

    return {
        "initial_decision": initial_decision,
        "post_test_decision": post_test_decision,
        "auto_test_results": auto_results,
    }


def run_cascade_test_phase(
    args: argparse.Namespace,
    winner: dict[str, Any],
    *,
    phase: str = "cascade_test",
) -> dict[str, Any]:
    test_run_name = f"{winner['run_name']}-test"
    existing = find_finished_run_across_experiments(
        args.mlflow_uri,
        (args.cascade_experiment_name, "Default"),
        run_name=test_run_name,
        phase=phase,
    )
    if existing and args.resume:
        return summary_from_existing(existing)

    experiment_id = configure_mlflow(args.mlflow_uri, args.cascade_experiment_name, args.mlflow_artifact_root)
    terminate_matching_runs(
        tracking_uri=args.mlflow_uri,
        experiment_name=args.cascade_experiment_name,
        run_name=phase,
        phase=phase,
        run_type="parent",
    )
    sample_rows = read_pair_manifest(args.test_csv)
    sample_path = sample_rows[0].noisy
    previous_handlers = install_termination_handlers()
    parent_run = mlflow.start_run(
        run_name=phase,
        experiment_id=experiment_id,
        tags={"phase": phase, "run_type": "parent"},
    )
    run_status = "FINISHED"
    sample_dir = Path("checkpoints/cascade") / f"{Path(test_run_name).name}_test_samples"
    try:
        child_run = mlflow.start_run(
            run_name=test_run_name,
            experiment_id=experiment_id,
            nested=True,
            tags={"phase": phase, "run_type": "child", "mlflow.parentRunId": parent_run.info.run_id},
        )
        child_status = "FINISHED"
        try:
            checkpoint_out = winner.get("checkpoint_out")
            checkpoint_path = Path(str(checkpoint_out)) if checkpoint_out else None
            if checkpoint_path and checkpoint_path.exists():
                model, package = load_model_from_checkpoint(
                    checkpoint_path,
                    device=args.device,
                    model_family=str(winner["model_family"]),
                    variant=str(winner["variant"]),
                )
                mlflow.log_artifact(checkpoint_path.as_posix(), artifact_path="checkpoints")
                mlflow.log_params(
                    {
                        "source_run_id": winner.get("run_id"),
                        "source_run_name": winner.get("run_name"),
                        "source_phase": winner.get("phase"),
                        "checkpoint_out": str(checkpoint_path),
                        "model_family": str(winner["model_family"]),
                        "variant": str(winner["variant"]),
                        "postfilter_mode": str(winner.get("postfilter_mode") or "none"),
                        "postfilter_preset": str(winner.get("postfilter_preset") or "medium"),
                        "train_postfilter": str(winner.get("train_postfilter")),
                        "spectral_native_gate": str(winner.get("spectral_native_gate")),
                    }
                )
                log_dict_artifact(
                    {
                        key: value
                        for key, value in package.items()
                        if key != "state_dict"
                    },
                    "reports/source_checkpoint_metadata.json",
                )
            else:
                model = build_enhancer(
                    str(winner["model_family"]),
                    str(winner["variant"]),
                    postfilter_mode=str(winner.get("postfilter_mode") or "none"),
                    postfilter_preset=str(winner.get("postfilter_preset") or "medium"),
                    train_postfilter=False,
                ).to(args.device)
                mlflow.log_params(
                    {
                        "source_run_id": winner.get("run_id"),
                        "source_run_name": winner.get("run_name"),
                        "source_phase": winner.get("phase"),
                        "pretrained_source": METRICGAN_PLUS_SOURCE,
                        "model_family": str(winner["model_family"]),
                        "variant": str(winner["variant"]),
                        "postfilter_mode": str(winner.get("postfilter_mode") or "none"),
                        "postfilter_preset": str(winner.get("postfilter_preset") or "medium"),
                    }
                )
            if sample_dir.exists():
                shutil.rmtree(sample_dir, ignore_errors=True)
            test_metrics = evaluate_manifest(
                model,
                args.test_csv,
                args.device,
                sample_rate=16000,
                compute_dnsmos=False,
                compute_composite=False,
                sample_dir=sample_dir,
                sample_count=3,
                batch_size=8,
                cache_audio=True,
            )
            latency_seconds = benchmark_inference(
                model,
                sample_path=sample_path,
                device=args.device,
                sample_rate=16000,
                duration_seconds=10,
                repeats=1,
            )
            mlflow.log_metrics(
                {
                    "test/pesq_mean": test_metrics["pesq_mean"],
                    "test/stoi_mean": test_metrics["stoi_mean"],
                    "test/sisdr_mean": test_metrics["sisdr_mean"],
                    "test/delta_snr_mean": test_metrics["delta_snr_mean"],
                    "test/inference_seconds_10s": latency_seconds,
                }
            )
            if sample_dir.exists():
                mlflow.log_artifacts(sample_dir.as_posix(), artifact_path="samples")
            log_dict_artifact(test_metrics, "reports/test_metrics.json")
            return {
                "run_id": child_run.info.run_id,
                "run_name": test_run_name,
                "winner_source": winner.get("run_name"),
                "test_metrics": test_metrics,
                "inference_seconds_10s": latency_seconds,
            }
        except KeyboardInterrupt:
            child_status = "KILLED"
            run_status = "KILLED"
            raise
        except BaseException:
            child_status = "FAILED"
            run_status = "FAILED"
            raise
        finally:
            if sample_dir.exists():
                shutil.rmtree(sample_dir, ignore_errors=True)
            mlflow.end_run(status=child_status)
    finally:
        restore_termination_handlers(previous_handlers)
        mlflow.end_run(status=run_status)


def promote_stage1_candidates(stage1_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in stage1_results:
        grouped[(str(result["model_family"]), str(result["source_run_name"]))].append(result)

    promoted: dict[str, dict[str, Any]] = {}
    for (family, _source_run_name), group in grouped.items():
        raw = next((item for item in group if item.get("postfilter_mode") == "none"), None)
        if raw is None or raw.get("best_val_select_pesq") is None or raw.get("best_val_select_stoi") is None:
            continue
        for candidate in group:
            if candidate.get("postfilter_mode") == "none":
                continue
            if candidate.get("best_val_select_pesq") is None or candidate.get("best_val_select_stoi") is None:
                continue
            delta_pesq = float(candidate["best_val_select_pesq"]) - float(raw["best_val_select_pesq"])
            delta_stoi = float(candidate["best_val_select_stoi"]) - float(raw["best_val_select_stoi"])
            if delta_pesq < 0.02 or delta_stoi < -0.01:
                continue
            enriched = dict(candidate)
            enriched["delta_pesq_vs_raw"] = delta_pesq
            enriched["delta_stoi_vs_raw"] = delta_stoi
            if family not in promoted or float(enriched["delta_pesq_vs_raw"]) > float(promoted[family]["delta_pesq_vs_raw"]):
                promoted[family] = enriched
    return list(promoted.values())


def gating_stage2_specs(
    args: argparse.Namespace,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    stage1_results: list[dict[str, Any]],
) -> list[ExperimentConfig]:
    specs: list[ExperimentConfig] = []
    for promoted in promote_stage1_candidates(stage1_results):
        for loss_recipe in ("R1", "R5", "R3", "R6"):
            for seed in (0, 1):
                specs.append(
                    ExperimentConfig(
                        train_csv=train_fit_csv,
                        val_rank_csv=val_rank_csv,
                        val_select_csv=val_select_csv,
                        checkpoint_out=(
                            f"checkpoints/gating/{promoted['model_family']}_{promoted['variant']}_"
                            f"{loss_recipe}_{promoted['postfilter_mode']}_{promoted['postfilter_preset']}_s{seed}.pt"
                        ),
                        model_family=str(promoted["model_family"]),
                        variant=str(promoted["variant"]),
                        loss_recipe=loss_recipe,
                        scheduler="plateau",
                        phase="gating_stage2",
                        epochs=args.epochs_gating_train,
                        lr=float(promoted["lr"]),
                        segment_len=int(promoted["segment_len"]),
                        seed=seed,
                        device=args.device,
                        mlflow_uri=args.mlflow_uri,
                        mlflow_artifact_root=args.mlflow_artifact_root,
                        experiment_name=args.gating_experiment_name,
                        early_stop_patience=args.gating_early_stop_patience,
                        min_epochs=args.gating_min_epochs,
                        lr_patience=4,
                        postfilter_mode=str(promoted["postfilter_mode"]),
                        postfilter_preset=str(promoted["postfilter_preset"]),
                        train_postfilter=True,
                    )
                )
    return specs


def best_wrapper_results(stage2_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for result in sorted(stage2_results, key=score_value, reverse=True):
        family = str(result["model_family"])
        if result.get("spectral_native_gate") in (True, "True"):
            continue
        if family not in chosen:
            chosen[family] = result
    return list(chosen.values())


def gating_stage3_specs(
    args: argparse.Namespace,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    stage2_results: list[dict[str, Any]],
) -> list[ExperimentConfig]:
    specs: list[ExperimentConfig] = []
    for winner in best_wrapper_results(stage2_results):
        if winner["model_family"] not in {"fullsubnet_plus", "mp_senet", "cmgan_small"}:
            continue
        for seed in (0, 1):
            specs.append(
                ExperimentConfig(
                    train_csv=train_fit_csv,
                    val_rank_csv=val_rank_csv,
                    val_select_csv=val_select_csv,
                    checkpoint_out=(
                        f"checkpoints/gating/{winner['model_family']}_{winner['variant']}_{winner['loss_recipe']}_"
                        f"native_{winner['postfilter_mode']}_{winner['postfilter_preset']}_s{seed}.pt"
                    ),
                    model_family=str(winner["model_family"]),
                    variant=str(winner["variant"]),
                    loss_recipe=str(winner["loss_recipe"]),
                    scheduler="plateau",
                    phase="gating_stage3",
                    epochs=args.epochs_gating_train,
                    lr=float(winner["lr"]),
                    segment_len=int(winner["segment_len"]),
                    seed=seed,
                    device=args.device,
                    mlflow_uri=args.mlflow_uri,
                    mlflow_artifact_root=args.mlflow_artifact_root,
                    experiment_name=args.gating_experiment_name,
                    early_stop_patience=args.gating_early_stop_patience,
                    min_epochs=args.gating_min_epochs,
                    lr_patience=4,
                    postfilter_mode=str(winner["postfilter_mode"]),
                    postfilter_preset=str(winner["postfilter_preset"]),
                    train_postfilter=True,
                    spectral_native_gate=True,
                )
            )
    return specs


def run_phase(
    phase: str,
    specs: list[ExperimentConfig],
    args: argparse.Namespace,
    *,
    experiment_name: str | None = None,
) -> list[dict[str, Any]]:
    campaign_log(f"starting {phase} with {len(specs)} specs in experiment {experiment_name or args.experiment_name}")
    experiment_name = experiment_name or args.experiment_name
    experiment_id = configure_mlflow(args.mlflow_uri, experiment_name, args.mlflow_artifact_root)
    terminate_matching_runs(
        tracking_uri=args.mlflow_uri,
        experiment_name=experiment_name,
        run_name=phase,
        phase=phase,
        run_type="parent",
    )
    results: list[dict[str, Any]] = []
    previous_handlers = install_termination_handlers()
    parent_run = mlflow.start_run(
        run_name=phase,
        experiment_id=experiment_id,
        tags={"phase": phase, "run_type": "parent"},
    )
    run_status = "FINISHED"
    try:
        limited_specs = maybe_limit(specs, args.max_runs)
        campaign_log(f"{phase}: executing {len(limited_specs)} specs after max-runs filter")
        for index, spec in enumerate(limited_specs, start=1):
            campaign_log(f"{phase}: [{index}/{len(limited_specs)}] {run_name(spec)}")
            spec.parent_run_id = parent_run.info.run_id
            results.append(run_or_resume(spec, resume=args.resume))
        log_dict_artifact({"phase": phase, "results": results}, f"campaign/{phase}_results.json")
        campaign_log(f"completed {phase} with {len(results)} results")
        return results
    except KeyboardInterrupt:
        run_status = "KILLED"
        campaign_log(f"{phase}: interrupted")
        raise
    except BaseException:
        run_status = "FAILED"
        campaign_log(f"{phase}: failed")
        raise
    finally:
        restore_termination_handlers(previous_handlers)
        mlflow.end_run(status=run_status)


def final_report(final_results: list[dict[str, Any]]) -> dict[str, Any]:
    test_pesq = [
        float(item["test_metrics"]["pesq_mean"])
        for item in final_results
        if item.get("test_metrics") and item["test_metrics"].get("pesq_mean") is not None
    ]
    test_stoi = [
        float(item["test_metrics"]["stoi_mean"])
        for item in final_results
        if item.get("test_metrics") and item["test_metrics"].get("stoi_mean") is not None
    ]
    test_sisdr = [
        float(item["test_metrics"]["sisdr_mean"])
        for item in final_results
        if item.get("test_metrics") and item["test_metrics"].get("sisdr_mean") is not None
    ]
    test_csig = [
        float(item["test_metrics"]["csig_mean"])
        for item in final_results
        if item.get("test_metrics") and item["test_metrics"].get("csig_mean") is not None
    ]
    test_cbak = [
        float(item["test_metrics"]["cbak_mean"])
        for item in final_results
        if item.get("test_metrics") and item["test_metrics"].get("cbak_mean") is not None
    ]
    test_covl = [
        float(item["test_metrics"]["covl_mean"])
        for item in final_results
        if item.get("test_metrics") and item["test_metrics"].get("covl_mean") is not None
    ]
    return {
        "test_pesq_mean": mean(test_pesq),
        "test_pesq_std": pstdev(test_pesq) if len(test_pesq) > 1 else 0.0,
        "test_stoi_mean": mean(test_stoi),
        "test_sisdr_mean": mean(test_sisdr),
        "test_csig_mean": mean(test_csig),
        "test_cbak_mean": mean(test_cbak),
        "test_covl_mean": mean(test_covl),
        "best_run": max(final_results, key=lambda item: float(item["test_metrics"]["pesq_mean"])),
    }


def run_stm32_teacher_cache_phase(
    args: argparse.Namespace,
    *,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
) -> dict[str, str]:
    experiment_id = configure_mlflow(args.mlflow_uri, args.stm32_experiment_name, args.mlflow_artifact_root)
    terminate_matching_runs(
        tracking_uri=args.mlflow_uri,
        experiment_name=args.stm32_experiment_name,
        run_name="stm32_teacher_cache",
        phase="stm32_teacher_cache",
    )
    teacher_run = _search_best_teacher_run(args)
    teacher_model = build_enhancer("metricgan_plus", "small").to(args.device)
    cache_root = Path(args.splits_dir) / f"stm32_teacher_cache_{args.stm32_audio_profile}"
    cache_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, str] = {}
    run = mlflow.start_run(
        run_name="stm32_teacher_cache",
        experiment_id=experiment_id,
        tags={"phase": "stm32_teacher_cache", "run_type": "parent"},
    )
    run_status = "FINISHED"
    try:
        mlflow.log_params(
            {
                "teacher_source_run_id": teacher_run["run_id"] if teacher_run else "metricgan_plus_pretrained",
                "teacher_source_run_name": teacher_run["run_name"] if teacher_run else "metricgan_plus-small-pretrained",
                "guidance_classic": "spectral_gating",
                "erb_bands": 32,
            }
        )
        manifests = {
            "train_fit": train_fit_csv,
            "val_rank": val_rank_csv,
            "val_select": val_select_csv,
        }
        for split_name, manifest_path in manifests.items():
            output_manifest = cache_root / split_name / f"{split_name}_teacher_cache.csv"
            if args.resume and output_manifest.exists():
                payload[split_name] = output_manifest.as_posix()
                continue
            payload[split_name] = build_teacher_cache(
                manifest_path,
                teacher_model,
                out_dir=cache_root / split_name,
                device=args.device,
                target_sample_rate=args.stm32_sample_rate,
                teacher_sample_rate=16000,
                erb_bands=32,
                guidance_classic="spectral_gating",
                progress_callback=lambda message, split_name=split_name: campaign_log(
                    f"stm32_teacher_cache[{split_name}]: {message}"
                ),
            )
        log_dict_artifact(payload, "campaign/stm32_teacher_cache.json")
        return payload
    except KeyboardInterrupt:
        run_status = "KILLED"
        raise
    except BaseException:
        run_status = "FAILED"
        raise
    finally:
        mlflow.end_run(status=run_status)


def run_stm32_classic_baseline_phase(args: argparse.Namespace, val_select_csv: str) -> dict[str, Any]:
    existing = find_finished_run(args.mlflow_uri, args.stm32_experiment_name, "stm32_classic_baseline", phase="stm32_classic_baseline")
    if existing and args.resume:
        existing_summary = stm32_result_from_existing(existing)
        has_current_shortlist = _mcu_shortlist_has_power_fields(existing_summary.get("mcu_shortlist"))
        has_current_summary = _stm32_summary_has_frequency_fields(existing_summary.get("stm32sim"))
        if has_current_shortlist and has_current_summary:
            return existing_summary
        stm32sim = simulate_classic_baseline(
            "spectral_gating",
            profile_name=args.stm32_profile,
            sample_rate=args.stm32_sample_rate,
            n_fft=args.stm32_n_fft,
            hop_length=args.stm32_hop_length,
            win_length=args.stm32_win_length,
            erb_bands=32,
        )
        shortlist_audit = simulate_baseline_across_profiles(
            "spectral_gating",
            shortlist_profiles=mcu_shortlist_profiles(args),
            reference_profiles=mcu_reference_profiles(args),
            sample_rate=args.stm32_sample_rate,
            n_fft=args.stm32_n_fft,
            hop_length=args.stm32_hop_length,
            win_length=args.stm32_win_length,
            erb_bands=32,
        )
        client = MlflowClient(tracking_uri=args.mlflow_uri)
        for key, value in _stm32_metric_fields(stm32sim).items():
            client.log_metric(existing_summary["run_id"], key, value)
        _log_shortlist_audit_to_run(args.mlflow_uri, existing_summary["run_id"], shortlist_audit)
        existing_summary["stm32sim"] = stm32sim
        existing_summary["mcu_shortlist"] = shortlist_audit
        return existing_summary

    experiment_id = configure_mlflow(args.mlflow_uri, args.stm32_experiment_name, args.mlflow_artifact_root)
    summary = summarize_classic_baselines(args.classic_baselines_xlsx)
    stm32sim = simulate_classic_baseline(
        "spectral_gating",
        profile_name=args.stm32_profile,
        sample_rate=args.stm32_sample_rate,
        n_fft=args.stm32_n_fft,
        hop_length=args.stm32_hop_length,
        win_length=args.stm32_win_length,
        erb_bands=32,
    )
    shortlist_audit = simulate_baseline_across_profiles(
        "spectral_gating",
        shortlist_profiles=mcu_shortlist_profiles(args),
        reference_profiles=mcu_reference_profiles(args),
        sample_rate=args.stm32_sample_rate,
        n_fft=args.stm32_n_fft,
        hop_length=args.stm32_hop_length,
        win_length=args.stm32_win_length,
        erb_bands=32,
    )
    baseline_model = SpectralGatingBaseline(
        n_fft=args.stm32_n_fft,
        hop_length=args.stm32_hop_length,
        win_length=args.stm32_win_length,
    ).to(args.device)
    campaign_log(
        "stm32_classic_baseline: evaluating spectral_gating on "
        f"{Path(val_select_csv).name} at {args.stm32_sample_rate} Hz"
    )
    baseline_metrics = evaluate_manifest(
        baseline_model,
        val_select_csv,
        args.device,
        sample_rate=args.stm32_sample_rate,
        compute_dnsmos=False,
        compute_composite=False,
        batch_size=8,
        cache_audio=True,
        progress_callback=lambda message: campaign_log(f"stm32_classic_baseline: {message}"),
    )
    campaign_log(
        "stm32_classic_baseline: completed evaluation with "
        f"PESQ={baseline_metrics['pesq_mean']:.4f}, STOI={baseline_metrics['stoi_mean']:.4f}"
    )
    spectral_gating_pesq = baseline_metrics["pesq_mean"]
    run = mlflow.start_run(
        run_name="stm32_classic_baseline",
        experiment_id=experiment_id,
        tags={"phase": "stm32_classic_baseline", "run_type": "parent"},
    )
    run_status = "FINISHED"
    try:
        mlflow.log_params(
            {
                "model_family": "spectral_gating",
                "variant": "classic",
                "phase": "stm32_classic_baseline",
                "classic_baselines_xlsx": args.classic_baselines_xlsx,
                "sample_rate": args.stm32_sample_rate,
                "n_fft": args.stm32_n_fft,
                "hop_length": args.stm32_hop_length,
                "win_length": args.stm32_win_length,
            }
        )
        mlflow.log_metrics(
            {
                "best/val_select_pesq_mean": float(spectral_gating_pesq or 0.0),
                "best/val_select_stoi_mean": float(baseline_metrics["stoi_mean"]),
                "best/val_select_sisdr_mean": float(baseline_metrics["sisdr_mean"]),
                "best/val_select_delta_snr_mean": float(baseline_metrics["delta_snr_mean"]),
                **_stm32_metric_fields(stm32sim),
                **_shortlist_metrics_from_audit(shortlist_audit),
            }
        )
        mlflow.set_tag("mcu_shortlist/supported_profiles", ",".join(shortlist_audit.get("supported_profiles") or []))
        mlflow.set_tag(
            "mcu_shortlist/hardware_supported_profiles",
            ",".join(shortlist_audit.get("hardware_supported_profiles") or []),
        )
        mlflow.set_tag(
            "mcu_shortlist/reference_supported_profiles",
            ",".join(shortlist_audit.get("reference_supported_profiles") or []),
        )
        mlflow.set_tag("mcu_shortlist/power_supported_profiles", ",".join(shortlist_audit.get("power_supported_profiles") or []))
        mlflow.set_tag(
            "mcu_shortlist/low_power_supported_profiles",
            ",".join(shortlist_audit.get("low_power_supported_profiles") or []),
        )
        if shortlist_audit.get("best_profile_name"):
            mlflow.set_tag("mcu_shortlist/best_profile_name", str(shortlist_audit["best_profile_name"]))
        if shortlist_audit.get("best_power_profile_name"):
            mlflow.set_tag("mcu_shortlist/best_power_profile_name", str(shortlist_audit["best_power_profile_name"]))
        if shortlist_audit.get("lowest_required_mhz_profile_name"):
            mlflow.set_tag("mcu_shortlist/lowest_required_mhz_profile_name", str(shortlist_audit["lowest_required_mhz_profile_name"]))
        if shortlist_audit.get("lowest_avg_power_profile_name"):
            mlflow.set_tag("mcu_shortlist/lowest_avg_power_profile_name", str(shortlist_audit["lowest_avg_power_profile_name"]))
        log_dict_artifact(
            {
                "classic_summary": summary,
                "baseline_metrics": baseline_metrics,
                "stm32sim": stm32sim,
                "mcu_shortlist": shortlist_audit,
            },
            "reports/stm32_classic_baseline.json",
        )
        return {
            "run_id": run.info.run_id,
            "run_name": "stm32_classic_baseline",
            "model_family": "spectral_gating",
            "variant": "classic",
            "phase": "stm32_classic_baseline",
            "best_val_select_pesq": spectral_gating_pesq,
            "best_val_select_stoi": baseline_metrics["stoi_mean"],
            "best_val_select_sisdr": baseline_metrics["sisdr_mean"],
            "stm32sim": stm32sim,
            "mcu_shortlist": shortlist_audit,
        }
    except KeyboardInterrupt:
        run_status = "KILLED"
        raise
    except BaseException:
        run_status = "FAILED"
        raise
    finally:
        mlflow.end_run(status=run_status)


def run_stm32_stage0_sim_phase(args: argparse.Namespace) -> list[dict[str, Any]]:
    experiment_id = configure_mlflow(args.mlflow_uri, args.stm32_experiment_name, args.mlflow_artifact_root)
    terminate_matching_runs(
        tracking_uri=args.mlflow_uri,
        experiment_name=args.stm32_experiment_name,
        run_name="stm32_stage0_sim",
        phase="stm32_stage0_sim",
        run_type="parent",
    )
    results: list[dict[str, Any]] = []
    parent_run = mlflow.start_run(
        run_name="stm32_stage0_sim",
        experiment_id=experiment_id,
        tags={"phase": "stm32_stage0_sim", "run_type": "parent"},
    )
    run_status = "FINISHED"
    try:
        teacher_audit = run_mcu_teacher_audit(args, _search_best_teacher_run(args))
        mlflow.log_metrics(
            {
                "teacher_direct/int8_supported_profile_count": float(teacher_audit["int8"].get("supported_profile_count") or 0),
                "teacher_direct/int8_hardware_supported_profile_count": float(
                    teacher_audit["int8"].get("hardware_supported_profile_count") or 0
                ),
                "teacher_direct/int8_power_supported_profile_count": float(
                    teacher_audit["int8"].get("power_supported_profile_count") or 0
                ),
                "teacher_direct/fp32_supported_profile_count": float(teacher_audit["fp32"].get("supported_profile_count") or 0),
            }
        )
        log_dict_artifact(teacher_audit, "reports/stm32_teacher_direct_audit.json")
        for spec in stm32_candidate_specs():
            run_name_local = _stm32_sim_run_name(spec)
            existing = find_finished_run(args.mlflow_uri, args.stm32_experiment_name, run_name_local, phase="stm32_stage0_sim")
            if existing and args.resume:
                existing_summary = stm32_result_from_existing(existing)
                has_current_shortlist = _mcu_shortlist_has_power_fields(existing_summary.get("mcu_shortlist"))
                has_current_summary = _stm32_summary_has_frequency_fields(existing_summary.get("stm32sim"))
                if has_current_shortlist and has_current_summary:
                    results.append(existing_summary)
                    continue
                model = build_enhancer(
                    spec["model_family"],
                    spec["variant"],
                    erb_bands=32,
                    context_frames=5,
                    guidance_classic=spec["guidance_classic"],
                    qat=False,
                    sample_rate=args.stm32_sample_rate,
                    n_fft=args.stm32_n_fft,
                    hop_length=args.stm32_hop_length,
                    win_length=args.stm32_win_length,
                )
                stm32sim = simulate_model_fit(model, profile_name=args.stm32_profile)
                shortlist_audit = simulate_model_across_profiles(
                    model,
                    shortlist_profiles=mcu_shortlist_profiles(args),
                    reference_profiles=mcu_reference_profiles(args),
                )
                client = MlflowClient(tracking_uri=args.mlflow_uri)
                for key, value in _stm32_metric_fields(stm32sim).items():
                    client.log_metric(existing_summary["run_id"], key, value)
                _log_shortlist_audit_to_run(args.mlflow_uri, existing_summary["run_id"], shortlist_audit)
                existing_summary["stm32sim"] = stm32sim
                existing_summary["mcu_shortlist"] = shortlist_audit
                results.append(existing_summary)
                continue
            model = build_enhancer(
                spec["model_family"],
                spec["variant"],
                erb_bands=32,
                context_frames=5,
                guidance_classic=spec["guidance_classic"],
                qat=False,
                sample_rate=args.stm32_sample_rate,
                n_fft=args.stm32_n_fft,
                hop_length=args.stm32_hop_length,
                win_length=args.stm32_win_length,
            )
            stm32sim = simulate_model_fit(model, profile_name=args.stm32_profile)
            shortlist_audit = simulate_model_across_profiles(
                model,
                shortlist_profiles=mcu_shortlist_profiles(args),
                reference_profiles=mcu_reference_profiles(args),
            )
            child = mlflow.start_run(
                run_name=run_name_local,
                experiment_id=experiment_id,
                nested=True,
                tags={"phase": "stm32_stage0_sim", "run_type": "child", "mlflow.parentRunId": parent_run.info.run_id},
            )
            mlflow.log_params(
                {
                    "model_family": spec["model_family"],
                    "variant": spec["variant"],
                    "guidance_classic": spec["guidance_classic"],
                    "phase": "stm32_stage0_sim",
                    "mcu_profile": args.stm32_profile,
                }
            )
            mlflow.log_metrics(
                {
                    **_stm32_metric_fields(stm32sim),
                    **_shortlist_metrics_from_audit(shortlist_audit),
                }
            )
            mlflow.set_tag("mcu_shortlist/supported_profiles", ",".join(shortlist_audit.get("supported_profiles") or []))
            mlflow.set_tag(
                "mcu_shortlist/hardware_supported_profiles",
                ",".join(shortlist_audit.get("hardware_supported_profiles") or []),
            )
            mlflow.set_tag(
                "mcu_shortlist/reference_supported_profiles",
                ",".join(shortlist_audit.get("reference_supported_profiles") or []),
            )
            mlflow.set_tag("mcu_shortlist/power_supported_profiles", ",".join(shortlist_audit.get("power_supported_profiles") or []))
            mlflow.set_tag(
                "mcu_shortlist/low_power_supported_profiles",
                ",".join(shortlist_audit.get("low_power_supported_profiles") or []),
            )
            if shortlist_audit.get("best_profile_name"):
                mlflow.set_tag("mcu_shortlist/best_profile_name", str(shortlist_audit["best_profile_name"]))
            if shortlist_audit.get("best_power_profile_name"):
                mlflow.set_tag("mcu_shortlist/best_power_profile_name", str(shortlist_audit["best_power_profile_name"]))
            if shortlist_audit.get("lowest_required_mhz_profile_name"):
                mlflow.set_tag("mcu_shortlist/lowest_required_mhz_profile_name", str(shortlist_audit["lowest_required_mhz_profile_name"]))
            if shortlist_audit.get("lowest_avg_power_profile_name"):
                mlflow.set_tag("mcu_shortlist/lowest_avg_power_profile_name", str(shortlist_audit["lowest_avg_power_profile_name"]))
            log_dict_artifact(
                {"stm32sim": stm32sim, "mcu_shortlist": shortlist_audit},
                f"reports/{run_name_local}_mcu_shortlist.json",
            )
            mlflow.end_run(status="FINISHED")
            result = {
                "run_id": child.info.run_id,
                "run_name": run_name_local,
                "model_family": spec["model_family"],
                "variant": spec["variant"],
                "guidance_classic": spec["guidance_classic"],
                "phase": "stm32_stage0_sim",
                "stm32sim": stm32sim,
                "mcu_shortlist": shortlist_audit,
            }
            results.append(result)
        log_dict_artifact({"results": results, "teacher_direct_audit": teacher_audit}, "campaign/stm32_stage0_sim.json")
        return results
    except KeyboardInterrupt:
        run_status = "KILLED"
        raise
    except BaseException:
        run_status = "FAILED"
        raise
    finally:
        mlflow.end_run(status=run_status)


def run_teacher_lite_stage0_sim_phase(args: argparse.Namespace) -> dict[str, Any]:
    experiment_id = configure_mlflow(args.mlflow_uri, args.teacher_lite_experiment_name, args.mlflow_artifact_root)
    terminate_matching_runs(
        tracking_uri=args.mlflow_uri,
        experiment_name=args.teacher_lite_experiment_name,
        run_name="teacher_lite_stage0_sim",
        phase="teacher_lite_stage0_sim",
        run_type="parent",
    )
    results: list[dict[str, Any]] = []
    teacher_model, teacher_source_summary = _load_teacher_lite_source_model(args)
    teacher_checkpoint, _ = _teacher_lite_teacher_checkpoint_path(args)
    teacher_sim_model, _ = load_model_from_checkpoint(
        teacher_checkpoint,
        device="cpu",
        model_family="metricgan_plus_native8k",
        variant="small",
    )
    teacher_as_is = simulate_model_across_profiles(
        teacher_sim_model,
        shortlist_profiles=mcu_shortlist_profiles(args),
        reference_profiles=mcu_reference_profiles(args),
        weight_bits=8,
    )
    parent_run = mlflow.start_run(
        run_name="teacher_lite_stage0_sim",
        experiment_id=experiment_id,
        tags={"phase": "teacher_lite_stage0_sim", "run_type": "parent"},
    )
    run_status = "FINISHED"
    try:
        mlflow.log_params(
            {
                "phase": "teacher_lite_stage0_sim",
                "teacher_variant": "metricgan_plus_native8k_int8",
                "teacher_lite_target_pesq": args.teacher_lite_target_pesq,
                "teacher_lite_latency_target_ms": 80.0,
            }
        )
        mlflow.log_metrics(
            {
                "teacher_as_is/power_supported_profile_count": float(teacher_as_is.get("power_supported_profile_count") or 0),
                "teacher_as_is/hardware_supported_profile_count": float(
                    teacher_as_is.get("hardware_supported_profile_count") or 0
                ),
                "teacher_as_is/low_power_supported_profile_count": float(
                    teacher_as_is.get("low_power_supported_profile_count") or 0
                ),
            }
        )
        log_dict_artifact(
            {
                "teacher_as_is_with_power_constraint": {
                    "supported_profiles": teacher_as_is.get("power_supported_profiles") or [],
                },
                "teacher_as_is_without_power_constraint": {
                    "hardware_supported_profiles": teacher_as_is.get("hardware_supported_profiles") or [],
                },
                "teacher_source": teacher_source_summary,
                "teacher_mcu_audit": teacher_as_is,
            },
            "reports/teacher_lite_teacher_as_is.json",
        )
        for spec in teacher_lite_candidate_specs():
            run_name_local = _teacher_lite_sim_run_name(spec)
            existing = find_finished_run(
                args.mlflow_uri,
                args.teacher_lite_experiment_name,
                run_name_local,
                phase="teacher_lite_stage0_sim",
            )
            if existing and args.resume:
                existing_summary = stm32_result_from_existing(existing)
                has_current_shortlist = _mcu_shortlist_has_power_fields(existing_summary.get("mcu_shortlist"))
                has_current_summary = _stm32_summary_has_frequency_fields(existing_summary.get("stm32sim"))
                if has_current_shortlist and has_current_summary:
                    results.append(existing_summary)
                    continue
            model = build_enhancer(
                spec["model_family"],
                spec["variant"],
                sample_rate=8000,
                n_fft=256,
                hop_length=80,
                win_length=160,
            )
            stm32sim = simulate_model_fit(model, profile_name=args.stm32_profile)
            shortlist_audit = simulate_model_across_profiles(
                model,
                shortlist_profiles=mcu_shortlist_profiles(args),
                reference_profiles=mcu_reference_profiles(args),
            )
            if existing and args.resume:
                existing_summary = stm32_result_from_existing(existing)
                client = MlflowClient(tracking_uri=args.mlflow_uri)
                for key, value in _stm32_metric_fields(stm32sim).items():
                    client.log_metric(existing_summary["run_id"], key, value)
                _log_shortlist_audit_to_run(args.mlflow_uri, existing_summary["run_id"], shortlist_audit)
                existing_summary["stm32sim"] = stm32sim
                existing_summary["mcu_shortlist"] = shortlist_audit
                results.append(existing_summary)
                continue
            child = mlflow.start_run(
                run_name=run_name_local,
                experiment_id=experiment_id,
                nested=True,
                tags={"phase": "teacher_lite_stage0_sim", "run_type": "child", "mlflow.parentRunId": parent_run.info.run_id},
            )
            mlflow.log_params(
                {
                    "model_family": spec["model_family"],
                    "variant": spec["variant"],
                    "phase": "teacher_lite_stage0_sim",
                    "mcu_profile": args.stm32_profile,
                    "sample_rate": 8000,
                    "n_fft": 256,
                    "hop_length": 80,
                    "win_length": 160,
                }
            )
            mlflow.log_metrics({**_stm32_metric_fields(stm32sim), **_shortlist_metrics_from_audit(shortlist_audit)})
            _log_shortlist_audit_to_run(args.mlflow_uri, child.info.run_id, shortlist_audit)
            log_dict_artifact({"stm32sim": stm32sim, "mcu_shortlist": shortlist_audit}, f"reports/{run_name_local}.json")
            mlflow.end_run(status="FINISHED")
            results.append(
                {
                    "run_id": child.info.run_id,
                    "run_name": run_name_local,
                    "model_family": spec["model_family"],
                    "variant": spec["variant"],
                    "phase": "teacher_lite_stage0_sim",
                    "stm32sim": stm32sim,
                    "mcu_shortlist": shortlist_audit,
                }
            )
        payload = {
            "results": results,
            "teacher_as_is": {
                "with_power_constraint": list(teacher_as_is.get("power_supported_profiles") or []),
                "without_power_constraint": list(teacher_as_is.get("hardware_supported_profiles") or []),
                "best_power_profile": teacher_as_is.get("best_power_profile_name"),
                "best_power_profile_avg_power_mw": teacher_as_is.get("best_power_profile_avg_power_mw"),
            },
            "teacher_source": teacher_source_summary,
        }
        log_dict_artifact(payload, "campaign/teacher_lite_stage0_sim.json")
        return payload
    except KeyboardInterrupt:
        run_status = "KILLED"
        raise
    except BaseException:
        run_status = "FAILED"
        raise
    finally:
        mlflow.end_run(status=run_status)


def ensure_teacher_lite_cache(
    args: argparse.Namespace,
    *,
    train_fit_csv: str,
) -> dict[str, Any]:
    teacher_model, teacher_summary = _load_teacher_lite_source_model(args)
    cache_root = Path(args.splits_dir) / "teacher_lite_cache_8k"
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_out = cache_root / "train_fit" / "train_fit_teacher_cache.csv"
    if args.resume and manifest_out.exists():
        return {
            "train_fit": manifest_out.as_posix(),
            "teacher_source": teacher_summary,
        }
    payload = build_teacher_cache(
        train_fit_csv,
        teacher_model,
        out_dir=cache_root / "train_fit",
        device="cpu",
        target_sample_rate=8000,
        teacher_sample_rate=8000,
        erb_bands=32,
        guidance_classic="none",
        progress_callback=lambda message: campaign_log(f"teacher_lite_cache[train_fit]: {message}"),
    )
    return {
        "train_fit": payload,
        "teacher_source": teacher_summary,
    }


def teacher_lite_stage1_specs(
    args: argparse.Namespace,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    teacher_cache: dict[str, Any],
    stage0_payload: dict[str, Any],
) -> list[ExperimentConfig]:
    stage0_results = list(stage0_payload.get("results") or [])
    allowed = {"metricgan_plus_native8k_causal_s", "metricgan_plus_native8k_causal_xs"}
    fit_lookup = {
        item["model_family"]: item
        for item in stage0_results
        if item.get("model_family") in allowed and _stm32_sim_is_eligible(item)
    }
    teacher_run = _teacher_lite_teacher_int8_run(args)
    specs: list[ExperimentConfig] = []
    for family in ("metricgan_plus_native8k_causal_s", "metricgan_plus_native8k_causal_xs"):
        if family not in fit_lookup:
            continue
        for seed in (0, 1):
            specs.append(
                ExperimentConfig(
                    train_csv=train_fit_csv,
                    val_rank_csv=val_rank_csv,
                    val_select_csv=val_select_csv,
                    checkpoint_out=f"checkpoints/teacher_lite/{family}_seed{seed}.pt",
                    model_family=family,
                    variant="small",
                    loss_recipe="D1",
                    phase="teacher_lite_stage1_train",
                    epochs=args.epochs_stm32_train,
                    lr=5e-4,
                    segment_len=16000,
                    seed=seed,
                    scheduler="plateau",
                    lr_patience=args.stm32_lr_patience,
                    early_stop_patience=args.stm32_early_stop_patience,
                    min_epochs=args.stm32_min_epochs,
                    eval_every=2,
                    eval_dnsmos=False,
                    rank_compute_composite=False,
                    select_compute_composite=False,
                    device=args.device,
                    mlflow_uri=args.mlflow_uri,
                    mlflow_artifact_root=args.mlflow_artifact_root,
                    experiment_name=args.teacher_lite_experiment_name,
                    teacher_source_run_id=teacher_run["run_id"] if teacher_run else None,
                    teacher_variant="metricgan_plus_native8k_int8",
                    teacher_cache_manifest=str(teacher_cache["train_fit"]),
                    guidance_classic="none",
                    erb_bands=32,
                    context_frames=5,
                    qat=False,
                    mcu_profile=args.stm32_profile,
                    sample_rate=8000,
                    n_fft=256,
                    hop_length=80,
                    win_length=160,
                )
            )
    return specs


def _teacher_lite_n6_stage_ready(stage0_payload: dict[str, Any]) -> bool:
    for item in stage0_payload.get("results") or []:
        if item.get("model_family") != "metricgan_plus_native8k_causal_n6":
            continue
        power_profiles = set(((item.get("mcu_shortlist") or {}).get("power_supported_profiles") or []))
        if power_profiles & LARGE_MCU_DEMO_PROFILE_NAMES:
            return True
    return False


def teacher_lite_n6_specs(
    args: argparse.Namespace,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    teacher_cache: dict[str, Any],
) -> list[ExperimentConfig]:
    teacher_run = _teacher_lite_teacher_int8_run(args)
    return [
        ExperimentConfig(
            train_csv=train_fit_csv,
            val_rank_csv=val_rank_csv,
            val_select_csv=val_select_csv,
            checkpoint_out="checkpoints/teacher_lite/metricgan_plus_native8k_causal_n6_seed0.pt",
            model_family="metricgan_plus_native8k_causal_n6",
            variant="small",
            loss_recipe="D1",
            phase="teacher_lite_stage1_train",
            epochs=args.epochs_stm32_train,
            lr=5e-4,
            segment_len=16000,
            seed=0,
            scheduler="plateau",
            lr_patience=args.stm32_lr_patience,
            early_stop_patience=args.stm32_early_stop_patience,
            min_epochs=args.stm32_min_epochs,
            eval_every=2,
            eval_dnsmos=False,
            rank_compute_composite=False,
            select_compute_composite=False,
            device=args.device,
            mlflow_uri=args.mlflow_uri,
            mlflow_artifact_root=args.mlflow_artifact_root,
            experiment_name=args.teacher_lite_experiment_name,
            teacher_source_run_id=teacher_run["run_id"] if teacher_run else None,
            teacher_variant="metricgan_plus_native8k_int8",
            teacher_cache_manifest=str(teacher_cache["train_fit"]),
            guidance_classic="none",
            erb_bands=32,
            context_frames=5,
            qat=False,
            mcu_profile=args.stm32_profile,
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            win_length=160,
        )
    ]


def choose_teacher_lite_stage1_action(
    stage1_results: list[dict[str, Any]],
    *,
    current_winner: dict[str, Any],
    classic_result: dict[str, Any],
    stage0_payload: dict[str, Any],
    target_pesq: float,
) -> dict[str, Any]:
    finished = [
        _with_stm32_recommendation(result)
        for result in stage1_results
        if result.get("best_val_select_pesq") is not None and _stm32_sim_is_eligible(result)
    ]
    if not finished:
        return {
            "next_action": "teacher_lite_stop",
            "reason": "no_finished_teacher_lite_stage1_runs",
            "winner": current_winner,
        }
    winner = max(finished, key=_stm32_candidate_sort_key)
    best_pesq = float(winner["best_val_select_pesq"])
    classic_pesq = float(classic_result.get("best_val_select_pesq") or 0.0)
    if best_pesq <= classic_pesq:
        return {
            "next_action": "teacher_lite_stop",
            "reason": "teacher_lite_below_spectral_gating",
            "winner": current_winner,
            "stage1_winner": winner,
        }
    if best_pesq > target_pesq:
        return {
            "next_action": "teacher_lite_qat",
            "reason": "teacher_lite_stage1_beat_current_winner",
            "winner": winner,
        }
    if _teacher_lite_n6_stage_ready(stage0_payload):
        return {
            "next_action": "teacher_lite_stage1_n6",
            "reason": "teacher_lite_small_variants_below_target",
            "winner": winner,
        }
    return {
        "next_action": "teacher_lite_stop",
        "reason": "teacher_lite_small_variants_below_target",
        "winner": current_winner,
        "stage1_winner": winner,
    }


def choose_teacher_lite_pre_qat_winner(
    stage1_results: list[dict[str, Any]],
    n6_results: list[dict[str, Any]],
    *,
    current_winner: dict[str, Any],
    classic_result: dict[str, Any],
    target_pesq: float,
) -> dict[str, Any]:
    valid = [
        _with_stm32_recommendation(result)
        for result in [*stage1_results, *n6_results]
        if result.get("best_val_select_pesq") is not None and _stm32_sim_is_eligible(result)
    ]
    if not valid:
        return {
            "next_action": "teacher_lite_stop",
            "reason": "teacher_lite_no_valid_pre_qat_winner",
            "winner": current_winner,
        }
    winner = max(valid, key=_stm32_candidate_sort_key)
    best_pesq = float(winner["best_val_select_pesq"])
    classic_pesq = float(classic_result.get("best_val_select_pesq") or 0.0)
    if best_pesq <= classic_pesq:
        return {
            "next_action": "teacher_lite_stop",
            "reason": "teacher_lite_expand_below_spectral_gating",
            "winner": current_winner,
            "pre_qat_winner": winner,
        }
    if best_pesq <= target_pesq:
        return {
            "next_action": "teacher_lite_stop",
            "reason": "teacher_lite_expand_below_current_winner",
            "winner": current_winner,
            "pre_qat_winner": winner,
        }
    return {
        "next_action": "teacher_lite_qat",
        "reason": "teacher_lite_expand_beat_current_winner",
        "winner": winner,
    }


def teacher_lite_qat_specs(
    args: argparse.Namespace,
    winner: dict[str, Any],
    teacher_cache: dict[str, Any],
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            train_csv=train_fit_csv,
            val_rank_csv=val_rank_csv,
            val_select_csv=val_select_csv,
            checkpoint_out=f"checkpoints/teacher_lite/{winner['model_family']}_qat.pt",
            model_family=str(winner["model_family"]),
            variant="small",
            loss_recipe="D2",
            phase="teacher_lite_qat",
            epochs=args.epochs_stm32_qat,
            lr=2e-4,
            segment_len=int(winner.get("segment_len") or 16000),
            seed=int(winner.get("seed") or 0),
            scheduler="plateau",
            lr_patience=args.stm32_lr_patience,
            early_stop_patience=max(4, args.stm32_early_stop_patience // 2),
            min_epochs=min(10, args.epochs_stm32_qat),
            eval_every=2,
            eval_dnsmos=False,
            rank_compute_composite=False,
            select_compute_composite=False,
            device=args.device,
            mlflow_uri=args.mlflow_uri,
            mlflow_artifact_root=args.mlflow_artifact_root,
            experiment_name=args.teacher_lite_experiment_name,
            teacher_source_run_id=winner.get("teacher_source_run_id"),
            teacher_variant="metricgan_plus_native8k_int8",
            teacher_cache_manifest=str(teacher_cache["train_fit"]),
            guidance_classic="none",
            erb_bands=32,
            context_frames=5,
            qat=True,
            mcu_profile=args.stm32_profile,
            init_checkpoint=str(winner["checkpoint_out"]),
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            win_length=160,
        )
    ]


def evaluate_teacher_lite_dynamic_quantized_candidate(
    args: argparse.Namespace,
    winner: dict[str, Any],
    *,
    val_select_csv: str,
) -> dict[str, Any]:
    model, _ = load_model_from_checkpoint(
        str(winner["checkpoint_out"]),
        device="cpu",
        model_family=str(winner["model_family"]),
        variant="small",
    )
    quantized = dynamic_quantize_metricgan(model)
    val_metrics = evaluate_manifest(
        quantized,
        val_select_csv,
        "cpu",
        sample_rate=8000,
        compute_dnsmos=False,
        compute_composite=False,
        batch_size=4,
        cache_audio=True,
        progress_callback=lambda message: campaign_log(f"teacher_lite_quant_bench: {message}"),
    )
    shortlist = simulate_model_across_profiles(
        model,
        shortlist_profiles=mcu_shortlist_profiles(args),
        reference_profiles=mcu_reference_profiles(args),
        weight_bits=8,
    )
    stm32sim = simulate_model_fit(model, profile_name=args.stm32_profile, weight_bits=8)
    return _with_stm32_recommendation(
        {
            "run_name": f"{winner['run_name']}-dynamic-int8",
            "model_family": winner["model_family"],
            "variant": "small",
            "phase": "teacher_lite_qat",
            "checkpoint_out": winner["checkpoint_out"],
            "best_val_select_pesq": val_metrics["pesq_mean"],
            "best_val_select_stoi": val_metrics["stoi_mean"],
            "best_val_select_sisdr": val_metrics["sisdr_mean"],
            "quantization_drop_pesq": float(winner["best_val_select_pesq"]) - float(val_metrics["pesq_mean"]),
            "stm32sim": stm32sim,
            "mcu_shortlist": shortlist,
        }
    )


def choose_teacher_lite_decision(
    *,
    current_winner: dict[str, Any],
    current_hardware_winner: dict[str, Any] | None,
    classic_result: dict[str, Any],
    stage0_payload: dict[str, Any],
    pre_qat_winner: dict[str, Any] | None,
    quantized_candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    teacher_as_is = stage0_payload.get("teacher_as_is") or {}
    current_hardware_winner = current_hardware_winner or current_winner

    def _stop_payload(reason: str) -> dict[str, Any]:
        return {
            "next_action": "teacher_lite_stop",
            "reason": reason,
            "low_power_direction": {
                "status": "keep_current_winner",
                "winner": current_winner,
                "reason": reason,
            },
            "max_quality_on_mcu_direction": {
                "status": "keep_current_winner",
                "winner": current_hardware_winner,
                "reason": reason,
            },
            "classic_baseline": classic_result,
            "teacher_as_is": teacher_as_is,
        }

    if pre_qat_winner is None or quantized_candidate is None:
        return _stop_payload("teacher_lite_no_quantized_candidate")
    if float(quantized_candidate.get("quantization_drop_pesq") or float("inf")) > 0.05:
        payload = _stop_payload("teacher_lite_quantization_drop_exceeds_limit")
        payload["pre_qat_winner"] = pre_qat_winner
        payload["quantized_candidate"] = quantized_candidate
        return payload
    low_power_ok = _stm32_sim_is_eligible(quantized_candidate, require_power=True)
    hardware_ok = _stm32_sim_is_eligible(quantized_candidate, require_power=False)
    low_power_beats_current = float(quantized_candidate.get("best_val_select_pesq") or 0.0) > float(
        current_winner.get("best_val_select_pesq") or 0.0
    )
    hardware_beats_current = float(quantized_candidate.get("best_val_select_pesq") or 0.0) > float(
        current_hardware_winner.get("best_val_select_pesq") or 0.0
    )

    low_power_direction = {
        "status": "accept" if low_power_ok and low_power_beats_current else "keep_current_winner",
        "winner": quantized_candidate if low_power_ok and low_power_beats_current else current_winner,
        "reason": (
            "teacher_lite_quantized_candidate_passed_low_power"
            if low_power_ok and low_power_beats_current
            else (
                "teacher_lite_quantized_candidate_failed_low_power_gate"
                if not low_power_ok
                else "teacher_lite_quantized_candidate_did_not_beat_current_low_power_winner"
            )
        ),
    }
    max_quality_direction = {
        "status": "accept" if hardware_ok and hardware_beats_current else "keep_current_winner",
        "winner": quantized_candidate if hardware_ok and hardware_beats_current else current_hardware_winner,
        "reason": (
            "teacher_lite_quantized_candidate_passed_hardware_gate"
            if hardware_ok and hardware_beats_current
            else (
                "teacher_lite_quantized_candidate_failed_hardware_gate"
                if not hardware_ok
                else "teacher_lite_quantized_candidate_did_not_beat_current_hardware_winner"
            )
        ),
    }

    if low_power_direction["status"] == "accept" and max_quality_direction["status"] == "accept":
        next_action = "teacher_lite_accept_both"
        reason = "teacher_lite_quantized_candidate_passed_both_directions"
    elif low_power_direction["status"] == "accept":
        next_action = "teacher_lite_accept_low_power"
        reason = "teacher_lite_quantized_candidate_passed_low_power_only"
    elif max_quality_direction["status"] == "accept":
        next_action = "teacher_lite_accept_max_quality_on_mcu"
        reason = "teacher_lite_quantized_candidate_passed_hardware_only"
    else:
        next_action = "teacher_lite_stop"
        reason = "teacher_lite_quantized_candidate_did_not_beat_current_winners"

    return {
        "next_action": next_action,
        "reason": reason,
        "winner": low_power_direction["winner"] if low_power_direction["status"] == "accept" else current_winner,
        "low_power_direction": low_power_direction,
        "max_quality_on_mcu_direction": max_quality_direction,
        "float_winner": pre_qat_winner,
        "quantized_candidate": quantized_candidate,
        "classic_baseline": classic_result,
        "teacher_as_is": teacher_as_is,
    }


def stm32_stage1_specs(
    args: argparse.Namespace,
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    teacher_cache: dict[str, str],
    stage0_results: list[dict[str, Any]],
) -> list[ExperimentConfig]:
    allowed_families = {"tiny_stm32_fc", "tiny_stm32_hybrid_sg"}
    fit_lookup = {
        item["model_family"]: item
        for item in stage0_results
        if item.get("model_family") in allowed_families
        and (item.get("stm32sim", {}) or {}).get("fit_ok")
        and (item.get("stm32sim", {}) or {}).get("realtime_ok")
    }
    teacher_run = _search_best_teacher_run(args)
    specs: list[ExperimentConfig] = []
    for family in ("tiny_stm32_fc", "tiny_stm32_hybrid_sg"):
        if family not in fit_lookup:
            continue
        guidance = "spectral_gating" if family != "tiny_stm32_fc" else "none"
        for seed in (0, 1):
            specs.append(
                ExperimentConfig(
                    train_csv=train_fit_csv,
                    val_rank_csv=val_rank_csv,
                    val_select_csv=val_select_csv,
                    checkpoint_out=f"checkpoints/stm32/{family}_seed{seed}.pt",
                    model_family=family,
                    variant="small",
                    loss_recipe="D1",
                    phase="stm32_stage1",
                    epochs=args.epochs_stm32_train,
                    lr=5e-4,
                    segment_len=args.stm32_segment_len,
                    seed=seed,
                    scheduler="plateau",
                    lr_patience=args.stm32_lr_patience,
                    early_stop_patience=args.stm32_early_stop_patience,
                    min_epochs=args.stm32_min_epochs,
                    eval_every=2,
                    eval_dnsmos=False,
                    rank_compute_composite=False,
                    select_compute_composite=False,
                    device=args.device,
                    mlflow_uri=args.mlflow_uri,
                    mlflow_artifact_root=args.mlflow_artifact_root,
                    experiment_name=args.stm32_experiment_name,
                    teacher_source_run_id=teacher_run["run_id"] if teacher_run else None,
                    teacher_cache_manifest=teacher_cache["train_fit"],
                    guidance_classic=guidance,
                    erb_bands=32,
                    context_frames=5,
                    qat=False,
                    mcu_profile=args.stm32_profile,
                    sample_rate=args.stm32_sample_rate,
                    n_fft=args.stm32_n_fft,
                    hop_length=args.stm32_hop_length,
                    win_length=args.stm32_win_length,
                )
            )
    return specs


def _stm32_sim_is_eligible(summary: dict[str, Any] | None, *, require_power: bool = True) -> bool:
    shortlist = (summary or {}).get("mcu_shortlist") or {}
    if shortlist:
        if require_power and "power_supported_profile_count" in shortlist:
            return int(shortlist.get("power_supported_profile_count") or 0) > 0
        if not require_power and "hardware_supported_profile_count" in shortlist:
            return int(shortlist.get("hardware_supported_profile_count") or 0) > 0
        if not require_power and "power_supported_profile_count" in shortlist:
            return int(shortlist.get("power_supported_profile_count") or 0) > 0
        return int(shortlist.get("supported_profile_count") or 0) > 0
    stm32sim = (summary or {}).get("stm32sim") or {}
    if "deployment_ok" in stm32sim:
        if require_power:
            return bool(stm32sim.get("deployment_ok"))
        return (
            bool(stm32sim.get("hardware_fit_ok", stm32sim.get("fit_ok")))
            and bool(stm32sim.get("frequency_ok"))
            and bool(stm32sim.get("hardware_realtime_ok", stm32sim.get("realtime_ok")))
            and (bool(stm32sim.get("latency_ok")) if "latency_ok" in stm32sim else True)
        )
    if "power_ok" in stm32sim or "frequency_ok" in stm32sim:
        frequency_ok = bool(stm32sim.get("frequency_ok")) if "frequency_ok" in stm32sim else True
        return (
            bool(stm32sim.get("hardware_fit_ok", stm32sim.get("fit_ok")))
            and frequency_ok
            and bool(stm32sim.get("hardware_realtime_ok", stm32sim.get("realtime_ok")))
            and (bool(stm32sim.get("power_ok")) if require_power else True)
            and (bool(stm32sim.get("latency_ok")) if "latency_ok" in stm32sim else True)
        )
    return bool(stm32sim.get("fit_ok")) and bool(stm32sim.get("realtime_ok"))


def attach_mcu_shortlist_audits(args: argparse.Namespace, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    client = MlflowClient(tracking_uri=args.mlflow_uri)
    for result in results:
        has_power_shortlist = _mcu_shortlist_has_power_fields(result.get("mcu_shortlist"))
        stm32sim_summary = (result.get("stm32sim") or {})
        has_power_summary = "avg_power_mw" in stm32sim_summary
        has_frequency_summary = "recommended_rt_mhz" in stm32sim_summary and "min_required_mhz" in stm32sim_summary
        has_macs_summary = "macs_per_hop_total" in stm32sim_summary
        if has_power_shortlist and has_power_summary and has_frequency_summary and has_macs_summary:
            continue
        checkpoint_path = result.get("checkpoint_out")
        if not checkpoint_path or not Path(str(checkpoint_path)).exists():
            continue
        model, _ = load_model_from_checkpoint(
            checkpoint_path,
            device="cpu",
            model_family=result.get("model_family"),
            variant=result.get("variant"),
        )
        stm32sim = simulate_model_fit(model, profile_name=str(result.get("mcu_profile") or args.stm32_profile))
        result["stm32sim"] = stm32sim
        audit = simulate_model_across_profiles(
            model,
            shortlist_profiles=mcu_shortlist_profiles(args),
            reference_profiles=mcu_reference_profiles(args),
        )
        result["mcu_shortlist"] = audit
        if result.get("run_id"):
            run_id = str(result["run_id"])
            for key, value in _stm32_metric_fields(stm32sim).items():
                client.log_metric(run_id, key, value)
            _log_shortlist_audit_to_run(args.mlflow_uri, run_id, audit)
    return results


def _stm32_stage0_lookup(stage0_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("model_family")): item
        for item in stage0_results
        if item.get("model_family")
    }


def build_stm32_family_strategy(
    stage0_results: list[dict[str, Any]],
    classic_result: dict[str, Any],
    threshold: float,
    *,
    sample_rate: int = 16000,
    teacher_direct_audit: dict[str, Any] | None = None,
    teacher_pesq: float | None = None,
    shortlist_profiles: Sequence[str] | None = None,
    reference_profiles: Sequence[str] | None = None,
    teacher_gap_max: float = 0.15,
) -> dict[str, Any]:
    classic_pesq = float(classic_result.get("best_val_select_pesq") or 0.0)
    stage0_lookup = _stm32_stage0_lookup(stage0_results)
    absolute_floor = 2.26 if sample_rate >= 16000 else 0.0
    return {
        "classic_baseline": {
            "name": "spectral_gating",
            "best_val_select_pesq": classic_pesq,
            "mcu_shortlist": classic_result.get("mcu_shortlist"),
        },
        "teacher_reference": {
            "test_pesq": teacher_pesq,
            "teacher_gap_max": teacher_gap_max,
            "direct_audit": teacher_direct_audit,
            "direct_viable_shortlist_profiles": list((teacher_direct_audit or {}).get("direct_viable_shortlist_profiles") or []),
            "direct_hardware_supported_shortlist_profiles": list(
                (teacher_direct_audit or {}).get("direct_hardware_supported_shortlist_profiles") or []
            ),
            "direct_power_supported_shortlist_profiles": list(
                (teacher_direct_audit or {}).get("direct_power_supported_shortlist_profiles") or []
            ),
            "direct_low_power_supported_shortlist_profiles": list(
                (teacher_direct_audit or {}).get("direct_low_power_supported_shortlist_profiles") or []
            ),
        },
        "hardware_shortlist": list(shortlist_profiles or []),
        "hardware_reference_profiles": list(reference_profiles or []),
        "promotion_thresholds": {
            "stage1_accept_min_pesq": absolute_floor,
            "stage1_accept_vs_classic": threshold,
            "stage1_accept_teacher_gap_max": teacher_gap_max,
            "expand_enable_vs_classic": max(threshold, 0.10),
            "expand_promote_vs_stage1": max(0.03, threshold / 2.0),
            "qat_max_drop_vs_float": 0.05,
            "power_budget_mw": 50.0,
            "frequency_guardband_ratio": 1.20,
        },
        "tiers": [
            {
                "phase": "stm32_stage1",
                "families": [
                    family
                    for family in ("tiny_stm32_fc", "tiny_stm32_hybrid_sg")
                    if _stm32_sim_is_eligible(stage0_lookup.get(family))
                ],
                "seeds": [0, 1],
                "goal": "Train the cheapest viable students first and choose the best float candidate.",
            },
            {
                "phase": "stm32_expand",
                "families": ["tiny_stm32_tcn_hybrid"] if _stm32_sim_is_eligible(stage0_lookup.get("tiny_stm32_tcn_hybrid")) else [],
                "seeds": [0, 1],
                "enable_if": f"stage1 winner beats spectral_gating by at least {max(threshold, 0.10):.2f} PESQ and passes stm32sim",
                "goal": "Probe the stronger hybrid only after tier-1 proves real value.",
            },
            {
                "phase": "stm32_qat",
                "families": ["winner_only"],
                "goal": "Quantization-aware tuning only on the single promoted winner.",
            },
            {
                "phase": "stm32_test",
                "families": ["winner_only"],
                "goal": "Holdout evaluation only for the accepted MCU candidate.",
            },
        ],
        "stage0_sim": [
            {
                "model_family": item.get("model_family"),
                "variant": item.get("variant"),
                "guidance_classic": item.get("guidance_classic"),
                "fit_ok": bool((item.get("stm32sim") or {}).get("fit_ok")),
                "frequency_ok": bool((item.get("stm32sim") or {}).get("frequency_ok")),
                "realtime_ok": bool((item.get("stm32sim") or {}).get("realtime_ok")),
                "power_ok": bool((item.get("stm32sim") or {}).get("power_ok")),
                "shortlist_supported_profile_count": int(((item.get("mcu_shortlist") or {}).get("supported_profile_count") or 0)),
                "shortlist_hardware_supported_profile_count": int(
                    ((item.get("mcu_shortlist") or {}).get("hardware_supported_profile_count") or 0)
                ),
                "shortlist_power_supported_profile_count": int(
                    ((item.get("mcu_shortlist") or {}).get("power_supported_profile_count") or 0)
                ),
                "shortlist_low_power_supported_profile_count": int(
                    ((item.get("mcu_shortlist") or {}).get("low_power_supported_profile_count") or 0)
                ),
                "ms_per_hop_80mhz": (item.get("stm32sim") or {}).get("ms_per_hop_80mhz"),
                "min_required_mhz": (item.get("stm32sim") or {}).get("min_required_mhz"),
                "recommended_rt_mhz": (item.get("stm32sim") or {}).get("recommended_rt_mhz"),
                "avg_power_mw": (item.get("stm32sim") or {}).get("avg_power_mw"),
                "flash_bytes": (item.get("stm32sim") or {}).get("flash_bytes"),
                "sram_peak_bytes": (item.get("stm32sim") or {}).get("sram_peak_bytes"),
                "recommended_profile_name": (item.get("mcu_shortlist") or {}).get("best_power_profile_name"),
            }
            for item in stage0_results
        ],
    }


def choose_stm32_stage1_winner(
    stage1_results: list[dict[str, Any]],
    classic_result: dict[str, Any],
    threshold: float,
    stage0_results: list[dict[str, Any]] | None = None,
    *,
    teacher_pesq: float | None = None,
    teacher_gap_max: float = 0.15,
    min_pesq: float = 2.26,
) -> dict[str, Any]:
    finished = [result for result in stage1_results if result.get("best_val_select_pesq") is not None]
    if not finished:
        return {
            "next_action": "stm32_stop",
            "winner": classic_result,
            "reason": "no_finished_stage1_runs",
        }
    winner = _with_stm32_recommendation(max(finished, key=_stm32_candidate_sort_key))
    winner_pesq = float(winner["best_val_select_pesq"])
    classic_pesq = float(classic_result.get("best_val_select_pesq") or 0.0)
    if not _stm32_sim_is_eligible(winner):
        return {
            "next_action": "stm32_stop",
            "winner": classic_result,
            "reason": "winner_failed_stm32sim",
        }
    if winner_pesq < min_pesq or winner_pesq < classic_pesq + threshold:
        return {
            "next_action": "stm32_stop",
            "winner": classic_result,
            "reason": "winner_below_classic_threshold",
            "stage1_winner": winner,
        }
    if teacher_pesq is not None and winner_pesq < float(teacher_pesq) - teacher_gap_max:
        return {
            "next_action": "stm32_stop",
            "winner": classic_result,
            "reason": "winner_below_teacher_gap_window",
            "stage1_winner": winner,
        }
    stage0_lookup = _stm32_stage0_lookup(stage0_results or [])
    expand_ready = _stm32_sim_is_eligible(stage0_lookup.get("tiny_stm32_tcn_hybrid"))
    if expand_ready and winner_pesq >= classic_pesq + max(threshold, 0.10):
        return {
            "next_action": "stm32_expand",
            "winner": winner,
            "reason": "stage1_winner_promoted_to_expand",
        }
    return {
        "next_action": "stm32_qat",
        "winner": winner,
        "reason": "stage1_winner_passed",
    }


def stm32_expand_specs(
    args: argparse.Namespace,
    winner: dict[str, Any],
    train_fit_csv: str,
    val_rank_csv: str,
    val_select_csv: str,
    teacher_cache: dict[str, str],
    stage0_results: list[dict[str, Any]],
) -> list[ExperimentConfig]:
    if not _stm32_sim_is_eligible(_stm32_stage0_lookup(stage0_results).get("tiny_stm32_tcn_hybrid")):
        return []
    teacher_run = _search_best_teacher_run(args)
    teacher_run_id = winner.get("teacher_source_run_id") or (teacher_run["run_id"] if teacher_run else None)
    specs: list[ExperimentConfig] = []
    for seed in (0, 1):
        specs.append(
            ExperimentConfig(
                train_csv=train_fit_csv,
                val_rank_csv=val_rank_csv,
                val_select_csv=val_select_csv,
                checkpoint_out=f"checkpoints/stm32/tiny_stm32_tcn_hybrid_expand_seed{seed}.pt",
                model_family="tiny_stm32_tcn_hybrid",
                variant="small",
                loss_recipe="D1",
                phase="stm32_expand",
                epochs=args.epochs_stm32_train,
                lr=5e-4,
                segment_len=args.stm32_segment_len,
                seed=seed,
                scheduler="plateau",
                lr_patience=args.stm32_lr_patience,
                early_stop_patience=args.stm32_early_stop_patience,
                min_epochs=args.stm32_min_epochs,
                eval_every=2,
                eval_dnsmos=False,
                rank_compute_composite=False,
                select_compute_composite=False,
                device=args.device,
                mlflow_uri=args.mlflow_uri,
                mlflow_artifact_root=args.mlflow_artifact_root,
                experiment_name=args.stm32_experiment_name,
                teacher_source_run_id=teacher_run_id,
                teacher_cache_manifest=teacher_cache["train_fit"],
                guidance_classic="spectral_gating",
                erb_bands=32,
                context_frames=5,
                qat=False,
                mcu_profile=args.stm32_profile,
                sample_rate=args.stm32_sample_rate,
                n_fft=args.stm32_n_fft,
                hop_length=args.stm32_hop_length,
                win_length=args.stm32_win_length,
            )
        )
    return specs


def choose_stm32_pre_qat_winner(
    classic_result: dict[str, Any],
    stage1_results: list[dict[str, Any]],
    expand_results: list[dict[str, Any]],
    threshold: float,
    *,
    teacher_pesq: float | None = None,
    teacher_gap_max: float = 0.15,
    min_pesq: float = 2.26,
) -> dict[str, Any]:
    initial = choose_stm32_stage1_winner(
        stage1_results,
        classic_result,
        threshold,
        teacher_pesq=teacher_pesq,
        teacher_gap_max=teacher_gap_max,
        min_pesq=min_pesq,
    )
    if initial["next_action"] == "stm32_stop":
        return initial

    stage1_winner = initial["winner"]
    if not expand_results:
        return {
            "next_action": "stm32_qat",
            "winner": stage1_winner,
            "reason": "stage1_winner_kept",
        }

    valid_expand = [
        result
        for result in expand_results
        if result.get("best_val_select_pesq") is not None and _stm32_sim_is_eligible(result)
    ]
    if not valid_expand:
        return {
            "next_action": "stm32_qat",
            "winner": stage1_winner,
            "reason": "expand_no_valid_runs",
            "stage1_winner": stage1_winner,
        }

    expand_winner = _with_stm32_recommendation(max(valid_expand, key=_stm32_candidate_sort_key))
    expand_pesq = float(expand_winner["best_val_select_pesq"])
    stage1_pesq = float(stage1_winner["best_val_select_pesq"])
    classic_pesq = float(classic_result.get("best_val_select_pesq") or 0.0)
    if expand_pesq >= max(stage1_pesq + max(0.03, threshold / 2.0), classic_pesq + threshold):
        if teacher_pesq is not None and expand_pesq < float(teacher_pesq) - teacher_gap_max:
            return {
                "next_action": "stm32_stop",
                "winner": classic_result,
                "reason": "expand_winner_below_teacher_gap_window",
                "stage1_winner": stage1_winner,
                "expand_winner": expand_winner,
            }
        return {
            "next_action": "stm32_qat",
            "winner": expand_winner,
            "reason": "expand_winner_promoted",
            "stage1_winner": stage1_winner,
            "expand_winner": expand_winner,
        }
    return {
        "next_action": "stm32_qat",
        "winner": stage1_winner,
        "reason": "stage1_winner_kept_over_expand",
        "stage1_winner": stage1_winner,
        "expand_winner": expand_winner,
    }


def stm32_qat_specs(args: argparse.Namespace, winner: dict[str, Any], train_fit_csv: str, val_rank_csv: str, val_select_csv: str, teacher_cache: dict[str, str]) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            train_csv=train_fit_csv,
            val_rank_csv=val_rank_csv,
            val_select_csv=val_select_csv,
            checkpoint_out=f"checkpoints/stm32/{winner['model_family']}_qat.pt",
            model_family=str(winner["model_family"]),
            variant=str(winner["variant"] or "small"),
            loss_recipe="D2",
            phase="stm32_qat",
            epochs=args.epochs_stm32_qat,
            lr=2e-4,
            segment_len=int(winner.get("segment_len") or 32000),
            seed=int(winner.get("seed") or 0),
            scheduler="plateau",
            lr_patience=args.stm32_lr_patience,
            early_stop_patience=max(4, args.stm32_early_stop_patience // 2),
            min_epochs=min(10, args.epochs_stm32_qat),
            eval_every=2,
            eval_dnsmos=False,
            rank_compute_composite=False,
            select_compute_composite=False,
            device=args.device,
            mlflow_uri=args.mlflow_uri,
            mlflow_artifact_root=args.mlflow_artifact_root,
            experiment_name=args.stm32_experiment_name,
            teacher_source_run_id=winner.get("teacher_source_run_id"),
            teacher_cache_manifest=teacher_cache["train_fit"],
            guidance_classic=str(winner.get("guidance_classic") or "none"),
            erb_bands=int(winner.get("erb_bands") or 32),
            context_frames=int(winner.get("context_frames") or 5),
            qat=True,
            mcu_profile=args.stm32_profile,
            init_checkpoint=str(winner["checkpoint_out"]),
            sample_rate=args.stm32_sample_rate,
            n_fft=args.stm32_n_fft,
            hop_length=args.stm32_hop_length,
            win_length=args.stm32_win_length,
        )
    ]


def choose_stm32_followup(
    classic_result: dict[str, Any],
    float_winner_decision: dict[str, Any],
    qat_results: list[dict[str, Any]],
    threshold: float,
    *,
    teacher_pesq: float | None = None,
    teacher_gap_max: float = 0.15,
) -> dict[str, Any]:
    if float_winner_decision["next_action"] != "stm32_qat":
        return float_winner_decision
    if not qat_results:
        return float_winner_decision
    valid_qat = [result for result in qat_results if result.get("best_val_select_pesq") is not None]
    if not valid_qat:
        return float_winner_decision
    qat_winner = _with_stm32_recommendation(max(valid_qat, key=_stm32_candidate_sort_key))
    stage1_winner = float_winner_decision["winner"]
    qat_pesq = float(qat_winner["best_val_select_pesq"])
    stage1_pesq = float(stage1_winner["best_val_select_pesq"])
    quantization_drop_pesq = stage1_pesq - qat_pesq
    if not _stm32_sim_is_eligible(qat_winner):
        return {
            "next_action": "stm32_stop",
            "winner": classic_result,
            "reason": "qat_failed_stm32sim",
            "stage1_winner": stage1_winner,
            "qat_winner": qat_winner,
            "quantization_drop_pesq": quantization_drop_pesq,
        }
    if quantization_drop_pesq > 0.05:
        return {
            "next_action": "stm32_stop",
            "winner": classic_result,
            "reason": "qat_drop_exceeds_limit",
            "stage1_winner": stage1_winner,
            "qat_winner": qat_winner,
            "quantization_drop_pesq": quantization_drop_pesq,
        }
    if teacher_pesq is not None and qat_pesq < float(teacher_pesq) - teacher_gap_max:
        return {
            "next_action": "stm32_stop",
            "winner": classic_result,
            "reason": "qat_below_teacher_gap_window",
            "stage1_winner": stage1_winner,
            "qat_winner": qat_winner,
            "quantization_drop_pesq": quantization_drop_pesq,
        }
    return {
        "next_action": "stm32_test",
        "winner": qat_winner,
        "reason": "qat_winner_passed",
        "stage1_winner": stage1_winner,
        "qat_winner": qat_winner,
        "quantization_drop_pesq": quantization_drop_pesq,
    }


def run_stm32_test_phase(args: argparse.Namespace, winner: dict[str, Any]) -> dict[str, Any]:
    config = ExperimentConfig(
        train_csv=args.train_csv,
        val_rank_csv=None,
        val_select_csv=None,
        test_csv=args.test_csv,
        checkpoint_out=str(winner["checkpoint_out"]),
        model_family=str(winner["model_family"]),
        variant=str(winner.get("variant") or "small"),
        loss_recipe=str(winner.get("loss_recipe") or "D2"),
        phase="stm32_test",
        epochs=1,
        batch_size=1,
        grad_accum=1,
        lr=0.0,
        segment_len=int(winner.get("segment_len") or 32000),
        seed=int(winner.get("seed") or 0),
        scheduler="plateau",
        eval_every=1,
        min_epochs=1,
        early_stop_patience=0,
        device=args.device,
        mlflow_uri=args.mlflow_uri,
        mlflow_artifact_root=args.mlflow_artifact_root,
        experiment_name=args.stm32_experiment_name,
        guidance_classic=str(winner.get("guidance_classic") or "none"),
        erb_bands=int(winner.get("erb_bands") or 32),
        context_frames=int(winner.get("context_frames") or 5),
        qat=bool(str(winner.get("qat")).lower() == "true"),
        mcu_profile=args.stm32_profile,
        sample_rate=args.stm32_sample_rate,
        n_fft=args.stm32_n_fft,
        hop_length=args.stm32_hop_length,
        win_length=args.stm32_win_length,
    )
    config.run_name = f"{winner['run_name']}-test"
    existing = find_finished_run(args.mlflow_uri, args.stm32_experiment_name, config.run_name, phase="stm32_test")
    if existing and args.resume:
        return stm32_result_from_existing(existing)
    experiment_id = configure_mlflow(args.mlflow_uri, args.stm32_experiment_name, args.mlflow_artifact_root)
    run = mlflow.start_run(
        run_name=config.run_name,
        experiment_id=experiment_id,
        tags={"phase": "stm32_test", "run_type": "child"},
    )
    run_status = "FINISHED"
    try:
        model, _ = load_model_from_checkpoint(
            config.checkpoint_out,
            device=config.device,
            model_family=config.model_family,
            variant=config.variant,
        )
        stm32sim = simulate_model_fit(model, profile_name=args.stm32_profile)
        shortlist_audit = simulate_model_across_profiles(
            model,
            shortlist_profiles=mcu_shortlist_profiles(args),
            reference_profiles=mcu_reference_profiles(args),
        )
        test_metrics = evaluate_manifest(
            model,
            args.test_csv,
            args.device,
            sample_rate=args.stm32_sample_rate,
            compute_dnsmos=False,
            compute_composite=False,
            batch_size=8,
            cache_audio=True,
            progress_callback=lambda message: campaign_log(f"stm32_test: {message}"),
        )
        mlflow.log_params(
            {
                "model_family": config.model_family,
                "variant": config.variant,
                "phase": "stm32_test",
                "guidance_classic": config.guidance_classic,
                "qat": config.qat,
                "mcu_profile": config.mcu_profile,
                "checkpoint_out": config.checkpoint_out,
            }
        )
        mlflow.log_metrics(
            {
                "test/pesq_mean": test_metrics["pesq_mean"],
                "test/stoi_mean": test_metrics["stoi_mean"],
                "test/sisdr_mean": test_metrics["sisdr_mean"],
                "test/delta_snr_mean": test_metrics["delta_snr_mean"],
                **_stm32_metric_fields(stm32sim),
                **_shortlist_metrics_from_audit(shortlist_audit),
            }
        )
        mlflow.set_tag("mcu_shortlist/supported_profiles", ",".join(shortlist_audit.get("supported_profiles") or []))
        mlflow.set_tag(
            "mcu_shortlist/hardware_supported_profiles",
            ",".join(shortlist_audit.get("hardware_supported_profiles") or []),
        )
        mlflow.set_tag(
            "mcu_shortlist/reference_supported_profiles",
            ",".join(shortlist_audit.get("reference_supported_profiles") or []),
        )
        mlflow.set_tag("mcu_shortlist/power_supported_profiles", ",".join(shortlist_audit.get("power_supported_profiles") or []))
        mlflow.set_tag(
            "mcu_shortlist/low_power_supported_profiles",
            ",".join(shortlist_audit.get("low_power_supported_profiles") or []),
        )
        if shortlist_audit.get("best_profile_name"):
            mlflow.set_tag("mcu_shortlist/best_profile_name", str(shortlist_audit["best_profile_name"]))
        if shortlist_audit.get("best_power_profile_name"):
            mlflow.set_tag("mcu_shortlist/best_power_profile_name", str(shortlist_audit["best_power_profile_name"]))
        if shortlist_audit.get("lowest_required_mhz_profile_name"):
            mlflow.set_tag("mcu_shortlist/lowest_required_mhz_profile_name", str(shortlist_audit["lowest_required_mhz_profile_name"]))
        if shortlist_audit.get("lowest_avg_power_profile_name"):
            mlflow.set_tag("mcu_shortlist/lowest_avg_power_profile_name", str(shortlist_audit["lowest_avg_power_profile_name"]))
        log_dict_artifact(
            {"test_metrics": test_metrics, "stm32sim": stm32sim, "mcu_shortlist": shortlist_audit},
            "reports/stm32_test.json",
        )
        return {
            "run_id": run.info.run_id,
            "run_name": config.run_name,
            "model_family": config.model_family,
            "variant": config.variant,
            "phase": "stm32_test",
            "checkpoint_out": config.checkpoint_out,
            "test_metrics": test_metrics,
            "stm32sim": stm32sim,
            "mcu_shortlist": shortlist_audit,
        }
    except KeyboardInterrupt:
        run_status = "KILLED"
        raise
    except BaseException:
        run_status = "FAILED"
        raise
    finally:
        mlflow.end_run(status=run_status)


def main() -> None:
    args = parse_args()
    teacher_audit_phases = {
        "teacher16k_fp32_ref",
        "teacher16k_int8_bench",
        "teacher8k_native_train",
        "teacher8k_native_int8_bench",
        "teacher_mcu_decision",
    }
    teacher_lite_phases = {
        "teacher_lite_stage0_sim",
        "teacher_lite_stage1_train",
        "teacher_lite_qat",
        "teacher_lite_decision",
    }
    if args.phase in teacher_lite_phases:
        args.stm32_audio_profile = "8k"
    if args.phase not in teacher_audit_phases:
        resolve_stm32_audio_args(args)
    args.device = require_cuda_device(args.device)
    campaign_log(f"launch phase={args.phase} device={args.device}")

    teacher_train_csv, teacher_test_csv, teacher_splits_dir = _teacher_reference_paths(args)
    split_train_csv = teacher_train_csv if args.phase in teacher_audit_phases else args.train_csv
    split_dir = teacher_splits_dir if args.phase in teacher_audit_phases else args.splits_dir

    campaign_log(f"building/reusing campaign splits from {split_dir}")
    split_paths = build_voicebank_campaign_splits(split_train_csv, split_dir)
    train_fit_csv = split_paths["train_fit"]
    val_rank_csv = split_paths["val_rank"]
    val_select_csv = split_paths["val_select"]
    campaign_log(
        f"splits ready train_fit={train_fit_csv} val_rank={val_rank_csv} val_select={val_select_csv}"
    )

    phase1_results: list[dict[str, Any]] = []
    phase2_results: list[dict[str, Any]] = []
    phase3_results: list[dict[str, Any]] = []
    gating_stage1_results: list[dict[str, Any]] = []
    gating_stage2_results: list[dict[str, Any]] = []
    cascade_stage1_results: list[dict[str, Any]] = []
    stm32_stage1_results: list[dict[str, Any]] = []
    stm32_expand_results: list[dict[str, Any]] = []
    stm32_qat_results: list[dict[str, Any]] = []

    if args.phase in teacher_audit_phases:
        teacher16k_ref = run_teacher16k_fp32_ref_phase(
            args,
            val_select_csv=val_select_csv,
            test_csv=teacher_test_csv,
        )
        if args.phase == "teacher16k_fp32_ref":
            print(json.dumps({"teacher16k_fp32_ref": teacher16k_ref}, indent=2, sort_keys=True, default=str))
            return

        teacher16k_int8 = run_teacher16k_int8_bench_phase(
            args,
            val_select_csv=val_select_csv,
            test_csv=teacher_test_csv,
            teacher16k_ref=teacher16k_ref,
        )
        if args.phase == "teacher16k_int8_bench":
            print(json.dumps({"teacher16k_int8_bench": teacher16k_int8}, indent=2, sort_keys=True, default=str))
            return

        teacher8k_fp32 = run_teacher8k_native_train_phase(
            args,
            train_fit_csv=train_fit_csv,
            val_rank_csv=val_rank_csv,
            val_select_csv=val_select_csv,
            test_csv=teacher_test_csv,
            teacher16k_ref=teacher16k_ref,
        )
        if args.phase == "teacher8k_native_train":
            print(json.dumps({"teacher8k_native_train": teacher8k_fp32}, indent=2, sort_keys=True, default=str))
            return

        teacher8k_drop = float(teacher8k_fp32.get("teacher_accuracy_drop_pesq") or float("inf"))
        teacher8k_int8: dict[str, Any] | None = None
        if teacher8k_drop <= 0.15:
            teacher8k_int8 = run_teacher8k_native_int8_bench_phase(
                args,
                val_select_csv=val_select_csv,
                test_csv=teacher_test_csv,
                teacher16k_ref=teacher16k_ref,
                teacher8k_fp32=teacher8k_fp32,
            )
        if args.phase == "teacher8k_native_int8_bench":
            print(
                json.dumps(
                    {"teacher8k_native_train": teacher8k_fp32, "teacher8k_native_int8_bench": teacher8k_int8},
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
            return

        decision = choose_teacher_mcu_decision(
            teacher16k_ref,
            teacher16k_int8,
            teacher8k_fp32,
            teacher8k_int8,
        )
        print(json.dumps({"teacher_mcu_decision": decision}, indent=2, sort_keys=True, default=str))
        return

    if args.phase in teacher_lite_phases:
        current_winner = _search_current_deploy_winner(args, require_power=True)
        current_hardware_winner = _search_current_deploy_winner(args, require_power=False)
        classic_result = _search_current_stm32_classic_result(args)
        teacher_cache: dict[str, Any] = {}
        if args.phase in {"teacher_lite_stage1_train", "teacher_lite_qat", "teacher_lite_decision"}:
            teacher_cache = ensure_teacher_lite_cache(
                args,
                train_fit_csv=train_fit_csv,
            )
        stage0_payload = run_teacher_lite_stage0_sim_phase(args)
        if args.phase == "teacher_lite_stage0_sim":
            print(
                json.dumps(
                    {
                        "teacher_lite_stage0_sim": stage0_payload,
                        "current_winner": current_winner,
                        "classic_baseline": classic_result,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
            return

        teacher_run = _teacher_lite_teacher_int8_run(args)
        teacher_pesq = None
        if teacher_run and teacher_run.get("test_metrics"):
            teacher_pesq = teacher_run["test_metrics"].get("pesq_mean")

        stage1_results = run_phase(
            "teacher_lite_stage1_train",
            teacher_lite_stage1_specs(args, train_fit_csv, val_rank_csv, val_select_csv, teacher_cache, stage0_payload),
            args,
            experiment_name=args.teacher_lite_experiment_name,
        )
        stage1_results = attach_mcu_shortlist_audits(args, stage1_results)
        annotate_stm32_gap_metrics(
            args.mlflow_uri,
            stage1_results,
            classic_pesq=float(classic_result.get("best_val_select_pesq") or 0.0),
            teacher_pesq=float(teacher_pesq) if teacher_pesq is not None else None,
        )
        if args.phase == "teacher_lite_stage1_train":
            print(
                json.dumps(
                    {
                        "teacher_lite_stage1_train": stage1_results,
                        "current_winner": current_winner,
                        "classic_baseline": classic_result,
                        "teacher_lite_stage0_sim": stage0_payload,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
            return

        stage1_action = choose_teacher_lite_stage1_action(
            stage1_results,
            current_winner=current_winner,
            classic_result=classic_result,
            stage0_payload=stage0_payload,
            target_pesq=args.teacher_lite_target_pesq,
        )

        n6_results: list[dict[str, Any]] = []
        if stage1_action["next_action"] == "teacher_lite_stage1_n6":
            n6_results = run_phase(
                "teacher_lite_stage1_train",
                teacher_lite_n6_specs(args, train_fit_csv, val_rank_csv, val_select_csv, teacher_cache),
                args,
                experiment_name=args.teacher_lite_experiment_name,
            )
            n6_results = attach_mcu_shortlist_audits(args, n6_results)
            annotate_stm32_gap_metrics(
                args.mlflow_uri,
                n6_results,
                classic_pesq=float(classic_result.get("best_val_select_pesq") or 0.0),
                teacher_pesq=float(teacher_pesq) if teacher_pesq is not None else None,
            )

        pre_qat_decision = stage1_action
        if stage1_action["next_action"] == "teacher_lite_stage1_n6":
            pre_qat_decision = choose_teacher_lite_pre_qat_winner(
                stage1_results,
                n6_results,
                current_winner=current_winner,
                classic_result=classic_result,
                target_pesq=args.teacher_lite_target_pesq,
            )
        if pre_qat_decision["next_action"] == "teacher_lite_stop":
            print(
                json.dumps(
                    {
                        "teacher_lite_decision": {
                            **pre_qat_decision,
                            "teacher_lite_stage0_sim": stage0_payload,
                            "current_winner": current_winner,
                            "classic_baseline": classic_result,
                        }
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
            return

        qat_results = run_phase(
            "teacher_lite_qat",
            teacher_lite_qat_specs(args, pre_qat_decision["winner"], teacher_cache, train_fit_csv, val_rank_csv, val_select_csv),
            args,
            experiment_name=args.teacher_lite_experiment_name,
        )
        qat_results = attach_mcu_shortlist_audits(args, qat_results)
        annotate_stm32_gap_metrics(
            args.mlflow_uri,
            qat_results,
            classic_pesq=float(classic_result.get("best_val_select_pesq") or 0.0),
            teacher_pesq=float(teacher_pesq) if teacher_pesq is not None else None,
        )
        if args.phase == "teacher_lite_qat":
            print(
                json.dumps(
                    {
                        "teacher_lite_qat": qat_results,
                        "pre_qat_winner": pre_qat_decision["winner"],
                        "teacher_lite_stage0_sim": stage0_payload,
                        "current_winner": current_winner,
                        "classic_baseline": classic_result,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
            return

        valid_qat = [
            _with_stm32_recommendation(result)
            for result in qat_results
            if result.get("best_val_select_pesq") is not None and _stm32_sim_is_eligible(result)
        ]
        qat_winner = max(valid_qat, key=_stm32_candidate_sort_key) if valid_qat else None
        quantized_candidate = (
            evaluate_teacher_lite_dynamic_quantized_candidate(args, qat_winner, val_select_csv=val_select_csv)
            if qat_winner is not None
            else None
        )
        decision = choose_teacher_lite_decision(
            current_winner=current_winner,
            current_hardware_winner=current_hardware_winner,
            classic_result=classic_result,
            stage0_payload=stage0_payload,
            pre_qat_winner=qat_winner,
            quantized_candidate=quantized_candidate,
        )
        print(
            json.dumps(
                {
                    "teacher_lite_decision": decision,
                    "teacher_lite_stage0_sim": stage0_payload,
                    "teacher_lite_stage1_train": stage1_results,
                    "teacher_lite_stage1_n6": n6_results,
                    "teacher_lite_qat": qat_results,
                    "current_winner": current_winner,
                    "current_hardware_winner": current_hardware_winner,
                    "classic_baseline": classic_result,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return

    if args.phase in {
        "stm32_teacher_cache",
        "stm32_classic_baseline",
        "stm32_stage0_sim",
        "stm32_stage1",
        "stm32_expand",
        "stm32_qat",
        "stm32_test",
        "stm32_auto",
    }:
        teacher_ref = _search_best_teacher_run(args)
        teacher_cache: dict[str, str] = {}
        stm32_strategy: dict[str, Any] | None = None
        teacher_direct_audit = run_mcu_teacher_audit(args, teacher_ref)

        if args.phase in {"stm32_teacher_cache", "stm32_stage1", "stm32_expand", "stm32_qat", "stm32_test", "stm32_auto"}:
            teacher_cache = run_stm32_teacher_cache_phase(
                args,
                train_fit_csv=train_fit_csv,
                val_rank_csv=val_rank_csv,
                val_select_csv=val_select_csv,
            )
            if args.phase == "stm32_teacher_cache":
                print(json.dumps({"teacher_cache": teacher_cache}, indent=2, sort_keys=True, default=str))
                return

        classic_result = run_stm32_classic_baseline_phase(args, val_select_csv)
        if args.phase == "stm32_classic_baseline":
            print(json.dumps({"classic_result": classic_result}, indent=2, sort_keys=True, default=str))
            return

        stage0_results = run_stm32_stage0_sim_phase(args)
        stm32_strategy = build_stm32_family_strategy(
            stage0_results,
            classic_result,
            args.stm32_improve_threshold,
            sample_rate=args.stm32_sample_rate,
            teacher_direct_audit=teacher_direct_audit,
            teacher_pesq=float(teacher_ref["test_pesq"]) if teacher_ref and teacher_ref.get("test_pesq") is not None else None,
            shortlist_profiles=mcu_shortlist_profiles(args),
            reference_profiles=mcu_reference_profiles(args),
            teacher_gap_max=args.stm32_teacher_gap_max,
        )
        if args.phase == "stm32_stage0_sim":
            print(
                json.dumps(
                    {"stage0_results": stage0_results, "stm32_strategy": stm32_strategy},
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
            return

        if args.phase in {"stm32_auto", "stm32_stage1", "stm32_expand", "stm32_qat", "stm32_test"}:
            if args.phase in {"stm32_auto", "stm32_stage1", "stm32_expand", "stm32_qat", "stm32_test"}:
                stm32_stage1_results = run_phase(
                    "stm32_stage1",
                    stm32_stage1_specs(args, train_fit_csv, val_rank_csv, val_select_csv, teacher_cache, stage0_results),
                    args,
                    experiment_name=args.stm32_experiment_name,
                )
                stm32_stage1_results = attach_mcu_shortlist_audits(args, stm32_stage1_results)
                annotate_stm32_gap_metrics(
                    args.mlflow_uri,
                    stm32_stage1_results,
                    classic_pesq=float(classic_result.get("best_val_select_pesq") or 0.0),
                    teacher_pesq=float(teacher_ref["test_pesq"]) if teacher_ref and teacher_ref.get("test_pesq") is not None else None,
                )
                if args.phase == "stm32_stage1":
                    print(
                        json.dumps(
                            {"stage1_results": stm32_stage1_results, "stm32_strategy": stm32_strategy},
                            indent=2,
                            sort_keys=True,
                            default=str,
                        )
                    )
                    return
            else:
                stm32_stage1_results = search_finished_results(args.mlflow_uri, args.stm32_experiment_name, phase="stm32_stage1")

            decision = choose_stm32_stage1_winner(
                stm32_stage1_results,
                classic_result,
                args.stm32_improve_threshold,
                stage0_results=stage0_results,
                teacher_pesq=float(teacher_ref["test_pesq"]) if teacher_ref and teacher_ref.get("test_pesq") is not None else None,
                teacher_gap_max=args.stm32_teacher_gap_max,
                min_pesq=2.26 if args.stm32_sample_rate >= 16000 else 0.0,
            )
            if decision["next_action"] == "stm32_stop":
                print(
                    json.dumps(
                        {"stm32_decision": decision, "stm32_strategy": stm32_strategy},
                        indent=2,
                        sort_keys=True,
                        default=str,
                    )
                )
                return

            float_winner_decision = decision
            if decision["next_action"] == "stm32_expand":
                stm32_expand_results = run_phase(
                    "stm32_expand",
                    stm32_expand_specs(
                        args,
                        decision["winner"],
                        train_fit_csv,
                        val_rank_csv,
                        val_select_csv,
                        teacher_cache,
                        stage0_results,
                    ),
                    args,
                    experiment_name=args.stm32_experiment_name,
                )
                stm32_expand_results = attach_mcu_shortlist_audits(args, stm32_expand_results)
                annotate_stm32_gap_metrics(
                    args.mlflow_uri,
                    stm32_expand_results,
                    classic_pesq=float(classic_result.get("best_val_select_pesq") or 0.0),
                    teacher_pesq=float(teacher_ref["test_pesq"]) if teacher_ref and teacher_ref.get("test_pesq") is not None else None,
                )
                float_winner_decision = choose_stm32_pre_qat_winner(
                    classic_result,
                    stm32_stage1_results,
                    stm32_expand_results,
                    args.stm32_improve_threshold,
                    teacher_pesq=float(teacher_ref["test_pesq"]) if teacher_ref and teacher_ref.get("test_pesq") is not None else None,
                    teacher_gap_max=args.stm32_teacher_gap_max,
                    min_pesq=2.26 if args.stm32_sample_rate >= 16000 else 0.0,
                )
                if args.phase == "stm32_expand":
                    print(
                        json.dumps(
                            {
                                "stage1_decision": decision,
                                "expand_results": stm32_expand_results,
                                "float_winner_decision": float_winner_decision,
                                "stm32_strategy": stm32_strategy,
                            },
                            indent=2,
                            sort_keys=True,
                            default=str,
                        )
                    )
                    return
            elif args.phase == "stm32_expand":
                print(
                    json.dumps(
                        {
                            "stage1_decision": decision,
                            "expand_results": [],
                            "float_winner_decision": {
                                "next_action": "stm32_skip_expand",
                                "winner": decision["winner"],
                                "reason": "expand_not_enabled",
                            },
                            "stm32_strategy": stm32_strategy,
                        },
                        indent=2,
                        sort_keys=True,
                        default=str,
                    )
                )
                return

            if args.phase in {"stm32_auto", "stm32_qat", "stm32_test"}:
                if args.phase in {"stm32_auto", "stm32_qat", "stm32_test"}:
                    stm32_qat_results = run_phase(
                        "stm32_qat",
                        stm32_qat_specs(args, float_winner_decision["winner"], train_fit_csv, val_rank_csv, val_select_csv, teacher_cache),
                        args,
                        experiment_name=args.stm32_experiment_name,
                    )
                    stm32_qat_results = attach_mcu_shortlist_audits(args, stm32_qat_results)
                    annotate_stm32_gap_metrics(
                        args.mlflow_uri,
                        stm32_qat_results,
                        classic_pesq=float(classic_result.get("best_val_select_pesq") or 0.0),
                        teacher_pesq=float(teacher_ref["test_pesq"]) if teacher_ref and teacher_ref.get("test_pesq") is not None else None,
                    )
                    if args.phase == "stm32_qat":
                        print(
                            json.dumps(
                                {
                                    "qat_results": stm32_qat_results,
                                    "float_winner_decision": float_winner_decision,
                                    "stm32_strategy": stm32_strategy,
                                },
                                indent=2,
                                sort_keys=True,
                                default=str,
                            )
                        )
                        return

                final_decision = choose_stm32_followup(
                    classic_result,
                    float_winner_decision,
                    stm32_qat_results,
                    args.stm32_improve_threshold,
                    teacher_pesq=float(teacher_ref["test_pesq"]) if teacher_ref and teacher_ref.get("test_pesq") is not None else None,
                    teacher_gap_max=args.stm32_teacher_gap_max,
                )
                if final_decision["next_action"] != "stm32_test":
                    print(
                        json.dumps(
                            {"stm32_decision": final_decision, "stm32_strategy": stm32_strategy},
                            indent=2,
                            sort_keys=True,
                            default=str,
                        )
                    )
                    return
                test_result = run_stm32_test_phase(args, final_decision["winner"])
                attach_mcu_shortlist_audits(args, [test_result])
                annotate_stm32_gap_metrics(
                    args.mlflow_uri,
                    [test_result],
                    classic_pesq=float(classic_result.get("best_val_select_pesq") or 0.0),
                    teacher_pesq=float(teacher_ref["test_pesq"]) if teacher_ref and teacher_ref.get("test_pesq") is not None else None,
                )
                print(
                    json.dumps(
                        {"stm32_decision": final_decision, "test_result": test_result, "stm32_strategy": stm32_strategy},
                        indent=2,
                        sort_keys=True,
                        default=str,
                    )
                )
                return

    if args.phase in {"gating_all", "gating_stage1", "gating_stage2", "gating_stage3"}:
        if args.phase in {"gating_all", "gating_stage1"}:
            gating_stage1_results = run_postfilter_phase(
                "gating_stage1",
                gating_stage1_specs(args, val_select_csv),
                args,
            )
            if args.phase == "gating_stage1":
                print(json.dumps({"stage1_results": gating_stage1_results}, indent=2, sort_keys=True, default=str))
                return
        else:
            gating_stage1_results = search_finished_results(args.mlflow_uri, args.gating_experiment_name, phase="gating_stage1")

        if args.phase in {"gating_all", "gating_stage2"}:
            gating_stage2_results = run_phase(
                "gating_stage2",
                gating_stage2_specs(args, train_fit_csv, val_rank_csv, val_select_csv, gating_stage1_results),
                args,
                experiment_name=args.gating_experiment_name,
            )
            if args.phase == "gating_stage2":
                print(json.dumps({"stage2_results": gating_stage2_results}, indent=2, sort_keys=True, default=str))
                return
        else:
            gating_stage2_results = search_finished_results(args.mlflow_uri, args.gating_experiment_name, phase="gating_stage2")

        if args.phase in {"gating_all", "gating_stage3"}:
            gating_stage3_results = run_phase(
                "gating_stage3",
                gating_stage3_specs(args, train_fit_csv, val_rank_csv, val_select_csv, gating_stage2_results),
                args,
                experiment_name=args.gating_experiment_name,
            )
            print(json.dumps({"stage3_results": gating_stage3_results}, indent=2, sort_keys=True, default=str))
            return

    if args.phase in {
        "cascade_all",
        "cascade_auto",
        "cascade_auto_next",
        "cascade_stage1",
        "cascade_stage2",
        "cascade_expand",
        "cascade_test",
    }:
        if args.phase in {"cascade_all", "cascade_stage1"}:
            cascade_stage1_results = run_cascade_stage1_phase(
                "cascade_stage1",
                cascade_stage1_specs(args, val_select_csv),
                args,
            )
            if args.phase == "cascade_stage1":
                print(json.dumps({"stage1_results": cascade_stage1_results}, indent=2, sort_keys=True, default=str))
                return
        else:
            cascade_stage1_results = search_finished_results_across_experiments(
                args.mlflow_uri,
                (args.cascade_experiment_name, "Default"),
                phase="cascade_stage1",
            )

        cascade_stage2_results: list[dict[str, Any]] = []
        if args.phase in {"cascade_all", "cascade_auto", "cascade_stage2"}:
            cascade_stage2_results = run_phase(
                "cascade_stage2",
                cascade_stage2_specs(args, train_fit_csv, val_rank_csv, val_select_csv, cascade_stage1_results),
                args,
                experiment_name=args.cascade_experiment_name,
            )
            if args.phase == "cascade_stage2":
                print(json.dumps({"stage2_results": cascade_stage2_results}, indent=2, sort_keys=True, default=str))
                return
        else:
            cascade_stage2_results = search_finished_results(
                args.mlflow_uri,
                args.cascade_experiment_name,
                phase="cascade_stage2",
            )

        if args.phase in {"cascade_all", "cascade_auto", "cascade_auto_next", "cascade_expand", "cascade_test"}:
            decision = choose_cascade_followup(args, cascade_stage1_results, cascade_stage2_results)
            campaign_log(
                "cascade follow-up decision: "
                f"winner={decision['winner']['run_name']} source={decision['winner_source']} "
                f"next_action={decision['next_action']} classic_top_pesq={decision['classic_summary'].get('top_pesq')}"
            )

            if decision["next_action"] == "cascade_expand" and args.phase in {"cascade_all", "cascade_auto", "cascade_expand"}:
                expand_results = run_phase(
                    "cascade_expand",
                    cascade_stage2_expand_specs(args, train_fit_csv, val_rank_csv, val_select_csv, cascade_stage1_results),
                    args,
                    experiment_name=args.cascade_experiment_name,
                )
                cascade_stage2_results = cascade_stage2_results + expand_results
                decision = choose_cascade_followup(args, cascade_stage1_results, cascade_stage2_results)
                campaign_log(
                    "cascade follow-up after expand: "
                    f"winner={decision['winner']['run_name']} source={decision['winner_source']} "
                    f"next_action={decision['next_action']}"
                )
                if decision["next_action"] == "cascade_expand":
                    decision["next_action"] = "cascade_stop"
                if args.phase == "cascade_expand":
                    print(
                        json.dumps(
                            {"cascade_decision": decision, "expand_results": expand_results},
                            indent=2,
                            sort_keys=True,
                            default=str,
                        )
                    )
                    return
            elif args.phase == "cascade_expand":
                print(json.dumps({"cascade_decision": decision, "expand_results": []}, indent=2, sort_keys=True, default=str))
                return

            if args.phase in {"cascade_all", "cascade_auto", "cascade_auto_next", "cascade_test"}:
                if decision["next_action"] == "cascade_stop":
                    print(json.dumps({"cascade_decision": decision}, indent=2, sort_keys=True, default=str))
                    return
                if args.phase == "cascade_test":
                    test_result = run_cascade_test_phase(args, decision["winner"])
                    print(
                        json.dumps(
                            {"cascade_decision": decision, "test_result": test_result},
                            indent=2,
                            sort_keys=True,
                            default=str,
                        )
                    )
                    return

                auto_next_payload = run_cascade_auto_next_phase(args, decision)
                print(json.dumps(auto_next_payload, indent=2, sort_keys=True, default=str))
                return

    if args.phase in {"all", "phase0"}:
        run_phase("phase0", phase0_specs(args, train_fit_csv, val_rank_csv, val_select_csv), args)
        if args.phase == "phase0":
            return

    if args.phase in {"all", "phase1"}:
        phase1_results = run_phase("phase1", phase1_specs(args, train_fit_csv, val_rank_csv, val_select_csv), args)
        if args.phase == "phase1":
            return
    else:
        phase1_results = []

    if args.phase in {"all", "phase2"}:
        if not phase1_results:
            phase1_results = run_phase("phase1", phase1_specs(args, train_fit_csv, val_rank_csv, val_select_csv), args)
        phase2_results = run_phase("phase2", phase2_specs(args, train_fit_csv, val_rank_csv, val_select_csv, phase1_results), args)
        if args.phase == "phase2":
            return
    else:
        phase2_results = []

    if args.phase in {"all", "phase3"}:
        if not phase2_results:
            if not phase1_results:
                phase1_results = run_phase("phase1", phase1_specs(args, train_fit_csv, val_rank_csv, val_select_csv), args)
            phase2_results = run_phase("phase2", phase2_specs(args, train_fit_csv, val_rank_csv, val_select_csv, phase1_results), args)
        phase3_results = run_phase("phase3", phase3_specs(args, train_fit_csv, val_rank_csv, val_select_csv, phase2_results), args)
        if args.phase == "phase3":
            return
    else:
        phase3_results = []

    if not phase3_results:
        if not phase2_results:
            if not phase1_results:
                phase1_results = run_phase("phase1", phase1_specs(args, train_fit_csv, val_rank_csv, val_select_csv), args)
            phase2_results = run_phase("phase2", phase2_specs(args, train_fit_csv, val_rank_csv, val_select_csv, phase1_results), args)
        phase3_results = run_phase("phase3", phase3_specs(args, train_fit_csv, val_rank_csv, val_select_csv, phase2_results), args)

    winner = select_phase3_winner(aggregate_phase3(phase3_results))

    if args.phase in {"all", "phase4"}:
        final_results = run_phase("phase4", phase4_specs(args, winner, args.train_csv, args.test_csv), args)
        report = final_report(final_results)
        register_run_model(
            tracking_uri=args.mlflow_uri,
            run_id=report["best_run"]["run_id"],
            model_name=args.registered_model_name,
            alias="best-pesq",
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
