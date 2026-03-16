from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torchaudio

from attenuate.audio import load_mono_audio, speaker_id_from_stem


def _infer_dirs(source_root: Path) -> dict[str, Path]:
    candidates = [
        {
            "train_clean": source_root / "clean_trainset_28spk_wav",
            "train_noisy": source_root / "noisy_trainset_28spk_wav",
            "test_clean": source_root / "clean_testset_wav",
            "test_noisy": source_root / "noisy_testset_wav",
        },
        {
            "train_clean": source_root / "train" / "clean",
            "train_noisy": source_root / "train" / "noisy",
            "test_clean": source_root / "test" / "clean",
            "test_noisy": source_root / "test" / "noisy",
        },
    ]
    for mapping in candidates:
        if all(path.exists() for path in mapping.values()):
            return mapping
    raise FileNotFoundError(
        "Could not infer VoiceBank directory layout from source root. "
        "Provide explicit --train-clean-dir/--train-noisy-dir/--test-clean-dir/--test-noisy-dir."
    )


def _pair_inventory(clean_dir: Path, noisy_dir: Path) -> dict[str, Any]:
    clean_map = {path.stem: path for path in clean_dir.glob("*.wav")}
    noisy_map = {path.stem: path for path in noisy_dir.glob("*.wav")}
    stems = sorted(clean_map.keys() & noisy_map.keys())
    if not stems:
        raise ValueError(f"No matched wav pairs between {clean_dir} and {noisy_dir}")
    return {
        "pairs": [(noisy_map[stem], clean_map[stem]) for stem in stems],
        "clean_only": sorted(clean_map.keys() - noisy_map.keys()),
        "noisy_only": sorted(noisy_map.keys() - clean_map.keys()),
    }


def _resample_copy(src: Path, dst: Path, sample_rate: int, overwrite: bool) -> Path:
    if dst.exists() and not overwrite:
        return dst
    wav, _ = load_mono_audio(src, sample_rate)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(dst.as_posix(), wav.unsqueeze(0), sample_rate)
    return dst


def _build_rows(
    pairs: Iterable[tuple[Path, Path]],
    out_root: Path,
    split: str,
    sample_rate: int,
    manifest_only: bool,
    overwrite: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for noisy_src, clean_src in pairs:
        if manifest_only:
            noisy_out = noisy_src
            clean_out = clean_src
        else:
            noisy_out = out_root / "audio" / split / "noisy" / noisy_src.name
            clean_out = out_root / "audio" / split / "clean" / clean_src.name
            _resample_copy(noisy_src, noisy_out, sample_rate, overwrite)
            _resample_copy(clean_src, clean_out, sample_rate, overwrite)
        rows.append({"noisy": noisy_out.as_posix(), "clean": clean_out.as_posix()})
    return rows


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["noisy", "clean"])
        writer.writeheader()
        writer.writerows(rows)


def _speaker_list(rows: list[dict[str, str]]) -> list[str]:
    return sorted({speaker_id_from_stem(Path(row["clean"]).stem) for row in rows})


def _sample_rate_hist(audio_dir: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in sorted(audio_dir.glob("*.wav")):
        info = torchaudio.info(path.as_posix())
        counts[str(int(info.sample_rate))] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _frame_mismatches(pairs: list[tuple[Path, Path]], limit: int = 32) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for noisy_path, clean_path in pairs:
        noisy_info = torchaudio.info(noisy_path.as_posix())
        clean_info = torchaudio.info(clean_path.as_posix())
        delta = int(noisy_info.num_frames) - int(clean_info.num_frames)
        if delta != 0:
            mismatches.append(
                {
                    "utterance_id": clean_path.stem,
                    "noisy_frames": int(noisy_info.num_frames),
                    "clean_frames": int(clean_info.num_frames),
                    "frame_delta": delta,
                }
            )
            if len(mismatches) >= limit:
                break
    return {"count": len(mismatches), "examples": mismatches}


def _split_train_val(
    rows: list[dict[str, str]],
    val_speaker_fraction: float,
    seed: int,
    allow_utterance_fallback: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    speaker_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        speaker_groups[speaker_id_from_stem(Path(row["clean"]).stem)].append(row)
    speakers = sorted(speaker_groups)

    if len(speakers) < 2:
        if not allow_utterance_fallback:
            raise ValueError("Need at least two speakers for speaker-disjoint val split")
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_speaker_fraction)))
        return shuffled[val_count:], shuffled[:val_count]

    rng = random.Random(seed)
    rng.shuffle(speakers)
    val_count = max(1, int(math.ceil(len(speakers) * val_speaker_fraction)))
    val_speakers = set(sorted(speakers[:val_count]))
    train_rows = [row for row in rows if speaker_id_from_stem(Path(row["clean"]).stem) not in val_speakers]
    val_rows = [row for row in rows if speaker_id_from_stem(Path(row["clean"]).stem) in val_speakers]
    return train_rows, val_rows


def _build_quick_subset(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if len(rows) <= limit:
        return list(rows)
    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: Path(item["clean"]).stem):
        by_speaker[speaker_id_from_stem(Path(row["clean"]).stem)].append(row)
    speakers = sorted(by_speaker)
    quick: list[dict[str, str]] = []
    while len(quick) < limit:
        progressed = False
        for speaker in speakers:
            if by_speaker[speaker]:
                quick.append(by_speaker[speaker].pop(0))
                progressed = True
                if len(quick) >= limit:
                    break
        if not progressed:
            break
    return quick


def prepare_voicebank_dataset(
    *,
    source_root: str | Path | None = None,
    train_clean_dir: str | Path | None = None,
    train_noisy_dir: str | Path | None = None,
    test_clean_dir: str | Path | None = None,
    test_noisy_dir: str | Path | None = None,
    out_root: str | Path,
    sample_rate: int = 16000,
    val_speaker_fraction: float = 0.1,
    val_quick_count: int = 96,
    seed: int = 1337,
    manifest_only: bool = False,
    overwrite: bool = False,
    allow_utterance_fallback: bool = False,
) -> dict[str, Any]:
    if source_root is None and not all([train_clean_dir, train_noisy_dir, test_clean_dir, test_noisy_dir]):
        raise ValueError("Provide either source_root or explicit train/test clean/noisy dirs")

    if source_root is not None:
        inferred = _infer_dirs(Path(source_root))
    else:
        inferred = {
            "train_clean": Path(train_clean_dir),
            "train_noisy": Path(train_noisy_dir),
            "test_clean": Path(test_clean_dir),
            "test_noisy": Path(test_noisy_dir),
        }

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_inventory = _pair_inventory(inferred["train_clean"], inferred["train_noisy"])
    test_inventory = _pair_inventory(inferred["test_clean"], inferred["test_noisy"])
    train_pairs = list(train_inventory["pairs"])
    test_pairs = list(test_inventory["pairs"])

    train_rows_all = _build_rows(
        train_pairs,
        out_root=out_root,
        split="train",
        sample_rate=sample_rate,
        manifest_only=manifest_only,
        overwrite=overwrite,
    )
    test_rows = _build_rows(
        test_pairs,
        out_root=out_root,
        split="test",
        sample_rate=sample_rate,
        manifest_only=manifest_only,
        overwrite=overwrite,
    )
    train_rows, val_rows = _split_train_val(
        train_rows_all,
        val_speaker_fraction=val_speaker_fraction,
        seed=seed,
        allow_utterance_fallback=allow_utterance_fallback,
    )
    val_quick_rows = _build_quick_subset(val_rows, val_quick_count)

    _write_manifest(out_root / "train.csv", train_rows)
    _write_manifest(out_root / "val.csv", val_rows)
    _write_manifest(out_root / "val_quick.csv", val_quick_rows)
    _write_manifest(out_root / "test.csv", test_rows)

    train_speakers = _speaker_list(train_rows)
    val_speakers = _speaker_list(val_rows)
    summary = {
        "paths": {key: value.as_posix() for key, value in inferred.items()},
        "out_root": out_root.as_posix(),
        "sample_rate": int(sample_rate),
        "manifest_only": bool(manifest_only),
        "splits": {
            "train": len(train_rows),
            "val": len(val_rows),
            "val_quick": len(val_quick_rows),
            "test": len(test_rows),
        },
        "speakers": {
            "train": train_speakers,
            "val": val_speakers,
            "speaker_disjoint": set(train_speakers).isdisjoint(val_speakers),
        },
        "pair_checks": {
            "train": {
                "matched_pairs": len(train_pairs),
                "clean_only_count": len(train_inventory["clean_only"]),
                "noisy_only_count": len(train_inventory["noisy_only"]),
                "clean_only_examples": train_inventory["clean_only"][:32],
                "noisy_only_examples": train_inventory["noisy_only"][:32],
                "frame_mismatches": _frame_mismatches(train_pairs),
            },
            "test": {
                "matched_pairs": len(test_pairs),
                "clean_only_count": len(test_inventory["clean_only"]),
                "noisy_only_count": len(test_inventory["noisy_only"]),
                "clean_only_examples": test_inventory["clean_only"][:32],
                "noisy_only_examples": test_inventory["noisy_only"][:32],
                "frame_mismatches": _frame_mismatches(test_pairs),
            },
        },
        "sample_rate_checks": {
            "train_clean": _sample_rate_hist(inferred["train_clean"]),
            "train_noisy": _sample_rate_hist(inferred["train_noisy"]),
            "test_clean": _sample_rate_hist(inferred["test_clean"]),
            "test_noisy": _sample_rate_hist(inferred["test_noisy"]),
        },
    }
    (out_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VoiceBank-DEMAND manifests at 16 kHz.")
    parser.add_argument("--source-root", type=str, default=None, help="Root with raw or pre-arranged VoiceBank audio.")
    parser.add_argument("--train-clean-dir", type=str, default=None)
    parser.add_argument("--train-noisy-dir", type=str, default=None)
    parser.add_argument("--test-clean-dir", type=str, default=None)
    parser.add_argument("--test-noisy-dir", type=str, default=None)
    parser.add_argument("--out-root", type=str, required=True, help="Output root for manifests and optionally audio.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--val-speaker-fraction", type=float, default=0.1)
    parser.add_argument("--val-quick-count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--manifest-only", action="store_true", help="Do not resample/copy audio; only write manifests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite resampled audio if it already exists.")
    parser.add_argument("--allow-utterance-fallback", action="store_true", help="Allow non speaker-disjoint split for smoke data.")
    args = parser.parse_args()

    summary = prepare_voicebank_dataset(
        source_root=args.source_root,
        train_clean_dir=args.train_clean_dir,
        train_noisy_dir=args.train_noisy_dir,
        test_clean_dir=args.test_clean_dir,
        test_noisy_dir=args.test_noisy_dir,
        out_root=args.out_root,
        sample_rate=args.sample_rate,
        val_speaker_fraction=args.val_speaker_fraction,
        val_quick_count=args.val_quick_count,
        seed=args.seed,
        manifest_only=args.manifest_only,
        overwrite=args.overwrite,
        allow_utterance_fallback=args.allow_utterance_fallback,
    )
    print(summary)


if __name__ == "__main__":
    main()
