from __future__ import annotations

import argparse
import json
from pathlib import Path

from sebench import MODEL_FAMILIES, MODEL_VARIANTS
from sebench.mlflow_utils import DEFAULT_ARTIFACT_ROOT, DEFAULT_EXPERIMENT_NAME, DEFAULT_TRACKING_URI
from sebench.runtime import require_cuda_device
from sebench.training import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a speech enhancement model with PESQ-aware validation logging.")
    parser.add_argument("--train-csv", required=True, help="CSV train split with columns noisy,clean.")
    parser.add_argument("--val-csv", default=None, help="Backward-compatible alias for --val-rank-csv.")
    parser.add_argument("--val-rank-csv", default=None, help="Fast validation split used for PESQ ranking.")
    parser.add_argument("--val-select-csv", default=None, help="Full internal validation split used for model selection.")
    parser.add_argument("--test-csv", default=None, help="Optional holdout manifest evaluated after training.")

    parser.add_argument("--model-family", default="atennuate", choices=list(MODEL_FAMILIES))
    parser.add_argument("--variant", default="base", choices=list(MODEL_VARIANTS))
    parser.add_argument("--loss-recipe", default="R1", choices=["R1", "R2", "R3", "R4", "R5", "R6", "D1", "D2"])
    parser.add_argument("--postfilter-mode", default="none", choices=["none", "sg_residual_soft", "sg_input_floor"])
    parser.add_argument("--postfilter-preset", default="medium", choices=["light", "medium", "aggressive"])
    parser.add_argument("--train-postfilter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spectral-native-gate", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--segment-len", type=int, default=32000)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--scheduler", default="plateau", choices=["plateau", "cosine"])
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--teacher-source-run-id", default=None)
    parser.add_argument("--teacher-cache-manifest", default=None)
    parser.add_argument("--guidance-classic", default="none", choices=["none", "spectral_gating"])
    parser.add_argument("--erb-bands", type=int, default=32)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--qat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quantize-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mcu-profile", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=320)

    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/experiment.pt")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--selection-metric", default="val_select_pesq_mean")
    parser.add_argument("--parent-run-id", default=None)
    parser.add_argument("--mlflow-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--mlflow-artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--log-system-metrics", action="store_true")
    parser.add_argument("--log-torch-model", action="store_true")
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--benchmark-seconds", type=int, default=10)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--max-eval-files", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--cache-eval-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rank-compute-composite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--select-compute-composite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-dnsmos", action="store_true")

    return parser.parse_args()


def namespace_to_config(args: argparse.Namespace) -> ExperimentConfig:
    val_rank_csv = args.val_rank_csv or args.val_csv
    device = require_cuda_device(args.device)

    return ExperimentConfig(
        train_csv=args.train_csv,
        val_rank_csv=val_rank_csv,
        val_select_csv=args.val_select_csv,
        test_csv=args.test_csv,
        checkpoint_out=args.checkpoint_out,
        model_family=args.model_family,
        variant=args.variant,
        loss_recipe=args.loss_recipe,
        run_name=args.run_name,
        phase=args.phase,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        segment_len=args.segment_len,
        num_workers=args.num_workers,
        scheduler=args.scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
        early_stop_patience=args.early_stop_patience,
        min_epochs=args.min_epochs,
        eval_every=args.eval_every,
        grad_clip=args.grad_clip,
        seed=args.seed,
        amp=args.amp,
        device=device,
        selection_metric=args.selection_metric,
        parent_run_id=args.parent_run_id,
        mlflow_uri=args.mlflow_uri,
        mlflow_artifact_root=args.mlflow_artifact_root,
        experiment_name=args.experiment_name,
        log_system_metrics=args.log_system_metrics,
        log_torch_model=args.log_torch_model,
        sample_count=args.sample_count,
        benchmark_seconds=args.benchmark_seconds,
        benchmark_repeats=args.benchmark_repeats,
        max_eval_files=args.max_eval_files,
        eval_batch_size=args.eval_batch_size,
        cache_eval_audio=args.cache_eval_audio,
        rank_compute_composite=args.rank_compute_composite,
        select_compute_composite=args.select_compute_composite,
        eval_dnsmos=(not args.no_dnsmos) and args.sample_rate == 16000,
        postfilter_mode=args.postfilter_mode,
        postfilter_preset=args.postfilter_preset,
        train_postfilter=args.train_postfilter,
        spectral_native_gate=args.spectral_native_gate,
        teacher_source_run_id=args.teacher_source_run_id,
        teacher_variant=None,
        audit_only=False,
        teacher_cache_manifest=args.teacher_cache_manifest,
        guidance_classic=args.guidance_classic,
        erb_bands=args.erb_bands,
        context_frames=args.context_frames,
        qat=args.qat,
        quantize_dynamic=args.quantize_dynamic,
        mcu_profile=args.mcu_profile,
        init_checkpoint=args.init_checkpoint,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )


def main() -> None:
    args = parse_args()
    if args.quantize_dynamic:
        raise ValueError(
            "--quantize-dynamic is evaluation-only for teacher audit benchmarks. "
            "Use campaign.py teacher audit phases or evaluate_metrics.py."
        )
    config = namespace_to_config(args)
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    checkpoint_path = Path(config.checkpoint_out)
    if checkpoint_path.exists():
        print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
