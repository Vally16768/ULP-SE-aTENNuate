#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as torch_f
import torchaudio

TARGET_SR = 16000
DEFAULT_OUTPUT_ROOT = Path("/mnt/ldm/ULP-SE-aTENNuate/dataset/dns5-headset-16k")
DEFAULT_STAGING_ROOT = Path("/mnt/ldm/DNS-Challenge")
DEFAULT_FREE_SPACE_FLOOR_GB = 50.0
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_MAX_AUGS = 1
DEFAULT_SNR_LOWER = -5.0
DEFAULT_SNR_UPPER = 20.0
DEFAULT_RIR_PROBABILITY = 0.5
DEFAULT_TARGET_DBFS = -20.0
SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".ogg"}

DNS5_BASE_URL = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"
FILELISTS_URL = "https://dnschallengepublic.blob.core.windows.net/dns5archive/filelists_headset.zip"
DEVTEST_URL = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_dev_testset.zip"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    blobs: tuple[str, ...] = ()
    url: str | None = None
    filelists: tuple[str, ...] = ()


CLEAN_SOURCE_ORDER = (
    "VocalSet_48kHz_mono",
    "emotional_speech",
    "vctk_wav48_silence_trimmed",
    "read_speech",
    "french_speech",
    "german_speech",
    "italian_speech",
    "russian_speech",
    "spanish_speech",
)


SOURCE_SPECS: Dict[str, SourceSpec] = {
    "VocalSet_48kHz_mono": SourceSpec(
        name="VocalSet_48kHz_mono",
        kind="clean",
        blobs=("Track1_Headset/VocalSet_48kHz_mono.tgz",),
        filelists=("english_vocalset.csv",),
    ),
    "emotional_speech": SourceSpec(
        name="emotional_speech",
        kind="clean",
        blobs=("Track1_Headset/emotional_speech.tgz",),
        filelists=("emotional_cremad.csv",),
    ),
    "french_speech": SourceSpec(
        name="french_speech",
        kind="clean",
        blobs=(
            "Track1_Headset/french_speech.tar.gz.partaa",
            "Track1_Headset/french_speech.tar.gz.partab",
            "Track1_Headset/french_speech.tar.gz.partac",
            "Track1_Headset/french_speech.tar.gz.partad",
            "Track1_Headset/french_speech.tar.gz.partae",
            "Track1_Headset/french_speech.tar.gz.partah",
        ),
        filelists=("french.csv",),
    ),
    "german_speech": SourceSpec(
        name="german_speech",
        kind="clean",
        blobs=(
            "Track1_Headset/german_speech.tgz.partaa",
            "Track1_Headset/german_speech.tgz.partab",
            "Track1_Headset/german_speech.tgz.partac",
            "Track1_Headset/german_speech.tgz.partad",
            "Track1_Headset/german_speech.tgz.partae",
            "Track1_Headset/german_speech.tgz.partaf",
            "Track1_Headset/german_speech.tgz.partag",
            "Track1_Headset/german_speech.tgz.partah",
            "Track1_Headset/german_speech.tgz.partaj",
            "Track1_Headset/german_speech.tgz.partal",
            "Track1_Headset/german_speech.tgz.partam",
            "Track1_Headset/german_speech.tgz.partan",
            "Track1_Headset/german_speech.tgz.partao",
            "Track1_Headset/german_speech.tgz.partap",
            "Track1_Headset/german_speech.tgz.partaq",
            "Track1_Headset/german_speech.tgz.partar",
            "Track1_Headset/german_speech.tgz.partas",
            "Track1_Headset/german_speech.tgz.partat",
            "Track1_Headset/german_speech.tgz.partau",
            "Track1_Headset/german_speech.tgz.partav",
            "Track1_Headset/german_speech.tgz.partaw",
        ),
        filelists=("german_mailabs.csv", "german_wikipedia.csv"),
    ),
    "italian_speech": SourceSpec(
        name="italian_speech",
        kind="clean",
        blobs=(
            "Track1_Headset/italian_speech.tgz.partaa",
            "Track1_Headset/italian_speech.tgz.partab",
            "Track1_Headset/italian_speech.tgz.partac",
            "Track1_Headset/italian_speech.tgz.partad",
        ),
        filelists=("italian.csv",),
    ),
    "read_speech": SourceSpec(
        name="read_speech",
        kind="clean",
        blobs=(
            "Track1_Headset/read_speech.tgz.partaa",
            "Track1_Headset/read_speech.tgz.partab",
            "Track1_Headset/read_speech.tgz.partac",
            "Track1_Headset/read_speech.tgz.partad",
            "Track1_Headset/read_speech.tgz.partae",
            "Track1_Headset/read_speech.tgz.partaf",
            "Track1_Headset/read_speech.tgz.partag",
            "Track1_Headset/read_speech.tgz.partah",
            "Track1_Headset/read_speech.tgz.partai",
            "Track1_Headset/read_speech.tgz.partaj",
            "Track1_Headset/read_speech.tgz.partak",
            "Track1_Headset/read_speech.tgz.partal",
            "Track1_Headset/read_speech.tgz.partam",
            "Track1_Headset/read_speech.tgz.partan",
            "Track1_Headset/read_speech.tgz.partao",
            "Track1_Headset/read_speech.tgz.partap",
            "Track1_Headset/read_speech.tgz.partaq",
            "Track1_Headset/read_speech.tgz.partar",
            "Track1_Headset/read_speech.tgz.partas",
            "Track1_Headset/read_speech.tgz.partat",
            "Track1_Headset/read_speech.tgz.partau",
        ),
        filelists=("english_read_speech.csv",),
    ),
    "russian_speech": SourceSpec(
        name="russian_speech",
        kind="clean",
        blobs=("Track1_Headset/russian_speech.tgz",),
        filelists=("russian.csv",),
    ),
    "spanish_speech": SourceSpec(
        name="spanish_speech",
        kind="clean",
        blobs=(
            "Track1_Headset/spanish_speech.tgz.partaa",
            "Track1_Headset/spanish_speech.tgz.partab",
            "Track1_Headset/spanish_speech.tgz.partac",
            "Track1_Headset/spanish_speech.tgz.partad",
            "Track1_Headset/spanish_speech.tgz.partae",
            "Track1_Headset/spanish_speech.tgz.partaf",
            "Track1_Headset/spanish_speech.tgz.partag",
        ),
        filelists=(
            "spanish_mailabs.csv",
            "spanish_slr39.csv",
            "spanish_slr61.csv",
            "spanish_slr71.csv",
            "spanish_slr73.csv",
            "spanish_slr74.csv",
            "spanish_slr75.csv",
        ),
    ),
    "vctk_wav48_silence_trimmed": SourceSpec(
        name="vctk_wav48_silence_trimmed",
        kind="clean",
        blobs=(
            "Track1_Headset/vctk_wav48_silence_trimmed.tgz.partaa",
            "Track1_Headset/vctk_wav48_silence_trimmed.tgz.partab",
            "Track1_Headset/vctk_wav48_silence_trimmed.tgz.partac",
        ),
        filelists=("english_vctk.csv",),
    ),
    "noise_ir": SourceSpec(
        name="noise_ir",
        kind="shared",
        blobs=(
            "noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2",
            "noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2",
            "noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2",
            "noise_fullband/datasets_fullband.noise_fullband.audioset_003.tar.bz2",
            "noise_fullband/datasets_fullband.noise_fullband.audioset_004.tar.bz2",
            "noise_fullband/datasets_fullband.noise_fullband.audioset_005.tar.bz2",
            "noise_fullband/datasets_fullband.noise_fullband.audioset_006.tar.bz2",
            "noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2",
            "noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2",
            "datasets_fullband.impulse_responses_000.tar.bz2",
        ),
    ),
    "filelists_headset": SourceSpec(
        name="filelists_headset",
        kind="metadata",
        url=FILELISTS_URL,
    ),
    "devtest": SourceSpec(
        name="devtest",
        kind="devtest",
        url=DEVTEST_URL,
    ),
}

SOURCE_ALIASES = {
    "smoke": (
        "VocalSet_48kHz_mono",
        "emotional_speech",
        "vctk_wav48_silence_trimmed",
        "noise_ir",
        "filelists_headset",
    ),
    "all_clean": CLEAN_SOURCE_ORDER,
    "all_relevant": CLEAN_SOURCE_ORDER + ("noise_ir", "filelists_headset", "devtest"),
    "shared": ("noise_ir", "filelists_headset", "devtest"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_seed(text: str) -> int:
    return int(stable_hex(text)[:16], 16)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_slashes(value: str) -> str:
    return value.replace("\\", "/").strip().strip("/")


def audio_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS
    )


def disk_free_gb(path: Path) -> float:
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(target)
    return usage.free / float(1024 ** 3)


def ensure_free_space(path: Path, floor_gb: float, context: str) -> None:
    free_gb = disk_free_gb(path)
    if free_gb < floor_gb:
        raise RuntimeError(
            f"Low disk space during {context}: free={free_gb:.2f} GB, floor={floor_gb:.2f} GB"
        )


def rel_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def run_command(cmd: Sequence[str] | str, *, shell: bool = False) -> None:
    subprocess.run(cmd, shell=shell, check=True)


def log(message: str) -> None:
    print(message, flush=True)


def state_dir(output_root: Path) -> Path:
    return ensure_dir(output_root / "state")


def state_file(output_root: Path, source: str, stage: str) -> Path:
    return state_dir(output_root) / f"{source}.{stage}.json"


def write_state(output_root: Path, source: str, stage: str, payload: dict) -> None:
    path = state_file(output_root, source, stage)
    data = {"source": source, "stage": stage, "updated_at_utc": utc_now(), **payload}
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_state(output_root: Path, source: str, stage: str) -> dict | None:
    path = state_file(output_root, source, stage)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def source_archives_root(staging_root: Path) -> Path:
    return ensure_dir(staging_root / "archives")


def source_extract_root(staging_root: Path) -> Path:
    return ensure_dir(staging_root / "extracted")


def clean_extract_dir(staging_root: Path, source: str) -> Path:
    return source_extract_root(staging_root) / "clean_sources" / source


def shared_extract_dir(staging_root: Path) -> Path:
    return ensure_dir(source_extract_root(staging_root) / "shared")


def filelists_extract_dir(staging_root: Path) -> Path:
    return source_extract_root(staging_root) / "filelists_headset"


def devtest_extract_dir(output_root: Path) -> Path:
    return ensure_dir(output_root / "devtest")


def shard_dir(output_root: Path) -> Path:
    return ensure_dir(output_root / "train_shards")


def combined_manifest_path(output_root: Path, split: str) -> Path:
    return output_root / f"{split}.csv"


def resolve_sources(items: Sequence[str]) -> List[SourceSpec]:
    resolved: List[SourceSpec] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name in SOURCE_ALIASES:
            for child in SOURCE_ALIASES[name]:
                add(child)
            return
        if name not in SOURCE_SPECS:
            raise ValueError(f"Unknown source/alias: {name}")
        if name in seen:
            return
        seen.add(name)
        resolved.append(SOURCE_SPECS[name])

    for item in items:
        add(item)
    return resolved


def archive_path_for_blob(staging_root: Path, blob: str) -> Path:
    return source_archives_root(staging_root) / blob


def metadata_archive_path(staging_root: Path, source: SourceSpec) -> Path:
    if source.name == "filelists_headset":
        return source_archives_root(staging_root) / "filelists_headset.zip"
    if source.name == "devtest":
        return source_archives_root(staging_root) / "V5_dev_testset.zip"
    raise ValueError(source.name)


def download_failure_root(staging_root: Path) -> Path:
    return ensure_dir(staging_root / ".download_failures")


def quarantine_bad_download(staging_root: Path, dest: Path, reason: str, expected_size: int) -> Path | None:
    if not dest.exists():
        return None
    batch = ensure_dir(
        download_failure_root(staging_root)
        / f"{dest.name}-{reason}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    moved_path = batch / dest.name
    shutil.move(str(dest), moved_path)
    metadata = {
        "reason": reason,
        "expected_size": expected_size,
        "actual_size": moved_path.stat().st_size,
        "url_name": dest.name,
        "quarantined_at_utc": utc_now(),
    }
    with (batch / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return moved_path


def remote_content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        content_length = response.headers.get("Content-Length")
        if not content_length:
            raise RuntimeError(f"Missing Content-Length for {url}")
        return int(content_length)


def download_file(url: str, dest: Path, staging_root: Path) -> None:
    ensure_dir(dest.parent)
    total_size = remote_content_length(url)
    tmp_body = dest.with_name(f".{dest.name}.range.tmp")
    while True:
        current_size = dest.stat().st_size if dest.exists() else 0
        if current_size > total_size:
            quarantined = quarantine_bad_download(staging_root, dest, "oversize", total_size)
            log(
                f"[download-quarantine] {dest.name}: "
                f"actual={current_size} expected={total_size} quarantined={quarantined}"
            )
            current_size = 0
        if current_size >= total_size:
            return
        try:
            tmp_body.unlink(missing_ok=True)
            result = subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "20",
                    "--speed-time",
                    "60",
                    "--speed-limit",
                    "1",
                    "--range",
                    f"{current_size}-{total_size - 1}",
                    "--output",
                    str(tmp_body),
                    "--write-out",
                    "%{http_code}",
                    url,
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            http_code = result.stdout.strip()
            if current_size > 0 and http_code != "206":
                raise RuntimeError(
                    f"Expected HTTP 206 for resumed download of {dest.name}, got {http_code}"
                )
            if current_size == 0 and http_code not in {"200", "206"}:
                raise RuntimeError(
                    f"Unexpected HTTP status for fresh download of {dest.name}: {http_code}"
                )
            if current_size == 0:
                shutil.move(str(tmp_body), str(dest))
            else:
                with tmp_body.open("rb") as in_f, dest.open("ab") as out_f:
                    shutil.copyfileobj(in_f, out_f, length=1024 * 1024)
                tmp_body.unlink(missing_ok=True)
            new_size = dest.stat().st_size
            if new_size > total_size:
                quarantined = quarantine_bad_download(
                    staging_root, dest, "oversize-post-append", total_size
                )
                raise RuntimeError(
                    f"Download overshot expected size for {dest.name}: "
                    f"actual={new_size} expected={total_size} quarantined={quarantined}"
                )
        except (
            OSError,
            subprocess.CalledProcessError,
            RuntimeError,
        ) as exc:
            tmp_body.unlink(missing_ok=True)
            resumed_size = dest.stat().st_size if dest.exists() else 0
            log(
                f"[download-retry] {dest.name}: "
                f"offset={resumed_size}/{total_size} error={exc}"
            )
            time.sleep(5)


def download_sources(
    sources: Sequence[SourceSpec],
    staging_root: Path,
    output_root: Path,
    free_space_floor_gb: float,
) -> None:
    for source in sources:
        ensure_free_space(staging_root, free_space_floor_gb, f"download:{source.name}")
        downloaded: List[str] = []
        if source.kind in {"clean", "shared"}:
            for blob in source.blobs:
                dest = archive_path_for_blob(staging_root, blob)
                url = f"{DNS5_BASE_URL}/{blob}"
                log(f"[download] {source.name}: {blob}")
                download_file(url, dest, staging_root)
                downloaded.append(rel_to(dest, staging_root))
        else:
            dest = metadata_archive_path(staging_root, source)
            log(f"[download] {source.name}: {source.url}")
            download_file(str(source.url), dest, staging_root)
            downloaded.append(rel_to(dest, staging_root))
        write_state(
            output_root,
            source.name,
            "download",
            {
                "files": downloaded,
                "count": len(downloaded),
            },
        )


def extract_multi_part_tar(parts: Sequence[Path], dest: Path) -> None:
    ensure_dir(dest)
    quoted = " ".join(f'"{str(p)}"' for p in parts)
    cmd = f"cat {quoted} | tar -xzf - -C \"{str(dest)}\""
    run_command(["bash", "-lc", cmd])


def extract_single_tar(archive: Path, dest: Path) -> None:
    ensure_dir(dest)
    suffix = archive.name.lower()
    if suffix.endswith(".tar.bz2"):
        run_command(["tar", "-xjf", str(archive), "-C", str(dest)])
    elif suffix.endswith(".tgz") or suffix.endswith(".tar.gz"):
        run_command(["tar", "-xzf", str(archive), "-C", str(dest)])
    else:
        raise ValueError(f"Unsupported tar archive type: {archive}")


def extract_devtest(archive: Path, output_root: Path) -> None:
    dest = devtest_extract_dir(output_root)
    ensure_dir(dest)
    with zipfile.ZipFile(archive) as zf:
        members = [
            info for info in zf.infolist()
            if normalize_slashes(info.filename).startswith("V5_dev_testset/Track1_Headset/")
        ]
        zf.extractall(dest, members=members)


def extract_sources(
    sources: Sequence[SourceSpec],
    staging_root: Path,
    output_root: Path,
    free_space_floor_gb: float,
) -> None:
    for source in sources:
        ensure_free_space(staging_root, free_space_floor_gb, f"extract:{source.name}")
        extracted: List[str] = []
        if source.kind == "clean":
            dest = clean_extract_dir(staging_root, source.name)
            ensure_dir(dest)
            archives = [archive_path_for_blob(staging_root, blob) for blob in source.blobs]
            for archive in archives:
                if not archive.exists():
                    raise FileNotFoundError(archive)
            if len(archives) == 1 and ".part" not in archives[0].name:
                extract_single_tar(archives[0], dest)
            elif len(archives) == 1:
                extract_multi_part_tar(archives, dest)
            else:
                extract_multi_part_tar(archives, dest)
            extracted.append(rel_to(dest, staging_root))
        elif source.name == "noise_ir":
            dest = shared_extract_dir(staging_root)
            ensure_dir(dest)
            for blob in source.blobs:
                archive = archive_path_for_blob(staging_root, blob)
                if not archive.exists():
                    raise FileNotFoundError(archive)
                extract_single_tar(archive, dest)
            extracted.append(rel_to(dest, staging_root))
        elif source.name == "filelists_headset":
            archive = metadata_archive_path(staging_root, source)
            dest = filelists_extract_dir(staging_root)
            ensure_dir(dest)
            run_command(["unzip", "-o", str(archive), "-d", str(dest)])
            extracted.append(rel_to(dest, staging_root))
        elif source.name == "devtest":
            archive = metadata_archive_path(staging_root, source)
            extract_devtest(archive, output_root)
            extracted.append(rel_to(devtest_extract_dir(output_root), output_root))
        else:
            raise ValueError(source.name)

        write_state(
            output_root,
            source.name,
            "extract",
            {
                "paths": extracted,
                "count": len(extracted),
            },
        )


def source_filelist_root(staging_root: Path) -> Path:
    base = filelists_extract_dir(staging_root)
    nested = base / "filelists_headset"
    if nested.exists():
        return nested
    return base


def source_path_candidates(source_name: str, value: str) -> List[str]:
    normalized = normalize_slashes(value).lower()
    if not normalized:
        return []
    candidates: List[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        candidate = candidate.strip().strip("/")
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    add(normalized)
    add(Path(normalized).name.lower())
    marker = f"{source_name.lower()}/"
    tail = normalized
    while marker in tail:
        tail = tail.split(marker, 1)[1].lstrip("/")
        add(tail)
    return candidates


def canonical_source_key(source_name: str, value: str) -> str:
    normalized = normalize_slashes(value).lower()
    if not normalized:
        return ""
    marker = f"{source_name.lower()}/"
    canonical = normalized
    while marker in canonical:
        canonical = canonical.split(marker, 1)[1].lstrip("/")
    return canonical or Path(normalized).name.lower()


def load_filelist_maps(staging_root: Path) -> tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    root = source_filelist_root(staging_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Missing extracted filelists at {root}. Run extract --source filelists_headset first."
        )

    speaker_by_source: Dict[str, Dict[str, str]] = {name: {} for name in CLEAN_SOURCE_ORDER}
    clean_key_by_source: Dict[str, Dict[str, str]] = {name: {} for name in CLEAN_SOURCE_ORDER}
    speaker_collisions: Dict[str, set[str]] = {name: set() for name in CLEAN_SOURCE_ORDER}
    clean_key_collisions: Dict[str, set[str]] = {name: set() for name in CLEAN_SOURCE_ORDER}

    for source_name in CLEAN_SOURCE_ORDER:
        source = SOURCE_SPECS[source_name]
        speaker_mapping = speaker_by_source[source_name]
        clean_key_mapping = clean_key_by_source[source_name]
        speaker_collision = speaker_collisions[source_name]
        clean_key_collision = clean_key_collisions[source_name]
        for csv_name in source.filelists:
            csv_path = root / csv_name
            if not csv_path.exists():
                continue
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_name = normalize_slashes(row.get("filename", ""))
                    speaker_id = row.get("speaker_id", "").strip()
                    if not raw_name or not speaker_id:
                        continue
                    candidates = source_path_candidates(source_name, raw_name)
                    canonical_key = canonical_source_key(source_name, raw_name)
                    for key in candidates:
                        if key not in speaker_collision:
                            existing = speaker_mapping.get(key)
                            if existing is None or existing == speaker_id:
                                speaker_mapping[key] = speaker_id
                            else:
                                speaker_collision.add(key)
                                speaker_mapping.pop(key, None)
                        if key in clean_key_collision:
                            continue
                        existing = clean_key_mapping.get(key)
                        if existing is None or existing == canonical_key:
                            clean_key_mapping[key] = canonical_key
                        else:
                            clean_key_collision.add(key)
                            clean_key_mapping.pop(key, None)
    return speaker_by_source, clean_key_by_source


def lookup_filelist_value(
    source: str,
    clean_path: Path,
    source_root: Path,
    lookup: Dict[str, Dict[str, str]],
) -> str | None:
    rel = clean_path.relative_to(source_root).as_posix()
    mapping = lookup.get(source, {})
    for key in source_path_candidates(source, rel):
        if key in mapping:
            return mapping[key]
    return None


def lookup_speaker_id(
    source: str,
    clean_path: Path,
    source_root: Path,
    lookup: Dict[str, Dict[str, str]],
) -> str | None:
    return lookup_filelist_value(source, clean_path, source_root, lookup)


def lookup_clean_key(
    source: str,
    clean_path: Path,
    source_root: Path,
    lookup: Dict[str, Dict[str, str]],
) -> str | None:
    return lookup_filelist_value(source, clean_path, source_root, lookup)


def split_name_for(source: str, key: str, val_fraction: float) -> str:
    bucket = stable_seed(f"{source}:{key}") % 10000
    return "val" if bucket < int(val_fraction * 10000) else "train"


def read_audio_full(path: Path, target_sr: int) -> torch.Tensor:
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Unreadable audio file {path}: {exc}") from exc
    mono = torch.from_numpy(np.mean(wav, axis=1, dtype=np.float32))
    if sr != target_sr:
        mono = torchaudio.functional.resample(mono, sr, target_sr)
    return mono.float().contiguous()


def read_audio_window(
    path: Path,
    target_sr: int,
    target_frames: int,
    rng: random.Random,
) -> torch.Tensor:
    try:
        info = sf.info(path)
    except Exception as exc:
        raise RuntimeError(f"Unreadable audio file {path}: {exc}") from exc
    approx_frames = int(math.ceil(target_frames * info.samplerate / target_sr))
    approx_frames = max(1, approx_frames + info.samplerate)
    max_offset = max(0, info.frames - approx_frames)
    start = 0 if max_offset == 0 else rng.randint(0, max_offset)
    try:
        wav, sr = sf.read(path, start=start, frames=approx_frames, dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Unreadable audio file {path}: {exc}") from exc
    mono = torch.from_numpy(np.mean(wav, axis=1, dtype=np.float32))
    if sr != target_sr:
        mono = torchaudio.functional.resample(mono, sr, target_sr)
    if mono.numel() == 0:
        mono = torch.zeros(target_frames, dtype=torch.float32)
    while mono.numel() < target_frames:
        mono = torch.cat([mono, mono], dim=0)
    if mono.numel() > target_frames:
        max_trim = mono.numel() - target_frames
        trim = 0 if max_trim == 0 else rng.randint(0, max_trim)
        mono = mono[trim:trim + target_frames]
    return mono.float().contiguous()


def peak_normalize(x: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    max_abs = float(torch.max(torch.abs(x)).item()) if x.numel() else 0.0
    if max_abs <= 1e-8:
        return x.clone()
    return (x / max_abs * peak).float().contiguous()


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.pow(2)) + 1e-8)


def apply_rir(clean: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    if rir.numel() == 0:
        return clean
    peak_idx = int(torch.argmax(torch.abs(rir)).item())
    rir = rir[peak_idx:]
    rir = rir / (torch.linalg.norm(rir, ord=2) + 1e-8)
    filt = torch.flip(rir, dims=[0]).view(1, 1, -1)
    signal = clean.view(1, 1, -1)
    padded = torch_f.pad(signal, (rir.numel() - 1, 0))
    out = torch_f.conv1d(padded, filt).view(-1)
    return out[:clean.numel()].float().contiguous()


def mix_clean_and_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    rng: random.Random,
    snr_lower: float,
    snr_upper: float,
) -> torch.Tensor:
    snr_db = rng.uniform(snr_lower, snr_upper)
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    scale = clean_rms / (noise_rms * (10 ** (snr_db / 20.0)))
    mixed = clean + noise * scale
    return mixed.float().contiguous()


def level_to_dbfs(x: torch.Tensor, target_dbfs: float) -> torch.Tensor:
    target_rms = 10 ** (target_dbfs / 20.0)
    current_rms = float(rms(x).item())
    if current_rms <= 1e-8:
        return x
    scaled = x * (target_rms / current_rms)
    peak = float(torch.max(torch.abs(scaled)).item()) if scaled.numel() else 0.0
    if peak > 0.99:
        scaled = scaled * (0.99 / peak)
    return scaled.float().contiguous()


def discover_noise_and_rir(staging_root: Path) -> tuple[List[Path], List[Path]]:
    shared = shared_extract_dir(staging_root)
    noise_dirs = [p for p in shared.rglob("*") if p.is_dir() and p.name == "noise_fullband"]
    rir_dirs = [p for p in shared.rglob("*") if p.is_dir() and p.name == "impulse_responses"]
    noise_files: List[Path] = []
    rir_files: List[Path] = []
    for noise_dir in noise_dirs:
        noise_files.extend(audio_files(noise_dir))
    for rir_dir in rir_dirs:
        rir_files.extend(audio_files(rir_dir))
    return sorted(set(noise_files)), sorted(set(rir_files))


def write_manifest(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["noisy", "clean"])
        writer.writeheader()
        writer.writerows(rows)


def manifest_lock_path(output_root: Path) -> Path:
    return state_dir(output_root) / "combine_manifests.lock"


def combine_manifests(output_root: Path) -> None:
    shard_root = shard_dir(output_root)
    lock_path = manifest_lock_path(output_root)
    ensure_dir(lock_path.parent)
    with lock_path.open("w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            for split in ("train", "val"):
                rows: List[dict] = []
                for csv_path in sorted(shard_root.glob(f"*.{split}.csv")):
                    with csv_path.open("r", newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        rows.extend(reader)
                rows.sort(key=lambda row: row["noisy"])
                write_manifest(combined_manifest_path(output_root, split), rows)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def selected_clean_sources(sources: Sequence[SourceSpec]) -> List[SourceSpec]:
    names = {source.name for source in sources if source.kind == "clean"}
    return [SOURCE_SPECS[name] for name in CLEAN_SOURCE_ORDER if name in names]


def output_pair_paths(
    output_root: Path,
    split: str,
    source: str,
    rel_clean_path: Path,
    aug_idx: int,
) -> tuple[Path, Path]:
    clean_rel = Path(source) / rel_clean_path.with_suffix(".wav")
    noisy_rel = Path(source) / rel_clean_path.with_suffix("")
    noisy_name = noisy_rel.name + f"__aug{aug_idx + 1:03d}.wav"
    clean_out = output_root / f"clean_{split}" / clean_rel
    noisy_out = output_root / f"noisy_{split}" / noisy_rel.parent / noisy_name
    return clean_out, noisy_out


def sample_valid_audio_window(
    candidates: Sequence[Path],
    *,
    target_sr: int,
    target_frames: int,
    rng: random.Random,
    bad_paths: set[Path],
    source_name: str,
    kind: str,
) -> torch.Tensor:
    seen_paths: set[Path] = set()
    max_random_attempts = min(len(candidates), 32)
    while len(seen_paths) < max_random_attempts:
        path = candidates[rng.randrange(len(candidates))]
        if path in bad_paths or path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            return peak_normalize(read_audio_window(path, target_sr, target_frames, rng))
        except Exception as exc:
            bad_paths.add(path)
            log(f"[warn] {source_name}: skipping unreadable {kind} file {path} ({exc})")
    for path in candidates:
        if path in bad_paths or path in seen_paths:
            continue
        try:
            return peak_normalize(read_audio_window(path, target_sr, target_frames, rng))
        except Exception as exc:
            bad_paths.add(path)
            log(f"[warn] {source_name}: skipping unreadable {kind} file {path} ({exc})")
    raise RuntimeError(f"No readable {kind} files available for {source_name}")


def sample_valid_audio_full(
    candidates: Sequence[Path],
    *,
    target_sr: int,
    rng: random.Random,
    bad_paths: set[Path],
    source_name: str,
    kind: str,
) -> torch.Tensor | None:
    seen_paths: set[Path] = set()
    max_random_attempts = min(len(candidates), 16)
    while len(seen_paths) < max_random_attempts:
        path = candidates[rng.randrange(len(candidates))]
        if path in bad_paths or path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            return peak_normalize(read_audio_full(path, target_sr))
        except Exception as exc:
            bad_paths.add(path)
            log(f"[warn] {source_name}: skipping unreadable {kind} file {path} ({exc})")
    for path in candidates:
        if path in bad_paths or path in seen_paths:
            continue
        try:
            return peak_normalize(read_audio_full(path, target_sr))
        except Exception as exc:
            bad_paths.add(path)
            log(f"[warn] {source_name}: skipping unreadable {kind} file {path} ({exc})")
    return None


def synthesize_source(
    source: SourceSpec,
    *,
    staging_root: Path,
    output_root: Path,
    free_space_floor_gb: float,
    max_augmentations_per_clean: int,
    val_fraction: float,
    limit_clean_files: int | None,
    snr_lower: float,
    snr_upper: float,
    rir_probability: float,
) -> None:
    clean_root = clean_extract_dir(staging_root, source.name)
    if not clean_root.exists():
        raise FileNotFoundError(
            f"Missing extracted source at {clean_root}. Run extract --source {source.name} first."
        )

    speaker_lookup, clean_key_lookup = load_filelist_maps(staging_root)
    extract_state = read_state(output_root, source.name, "extract") or {}
    allow_salvage_fallback = extract_state.get("extract") == "partial-salvage-merged-stashes"
    noise_files, rir_files = discover_noise_and_rir(staging_root)
    if not noise_files:
        raise RuntimeError("No extracted noise files found. Run extract --source noise_ir first.")

    discovered_clean_files = audio_files(clean_root)
    if limit_clean_files is not None:
        discovered_clean_files = discovered_clean_files[:limit_clean_files]
    if not discovered_clean_files:
        raise RuntimeError(f"No clean audio files found under {clean_root}")
    clean_items: List[tuple[Path, str]] = []
    skipped_unlisted_clean_files: List[str] = []
    duplicate_clean_files: List[str] = []
    seen_clean_keys: set[str] = set()
    for clean_path in discovered_clean_files:
        rel_clean = clean_path.relative_to(clean_root)
        clean_key = lookup_clean_key(source.name, clean_path, clean_root, clean_key_lookup)
        if clean_key is None and allow_salvage_fallback:
            clean_key = canonical_source_key(source.name, rel_clean.as_posix())
        if clean_key is None:
            skipped_unlisted_clean_files.append(clean_path.resolve().as_posix())
            continue
        if clean_key in seen_clean_keys:
            duplicate_clean_files.append(clean_path.resolve().as_posix())
            continue
        seen_clean_keys.add(clean_key)
        clean_items.append((clean_path, clean_key))
    if not clean_items:
        raise RuntimeError(f"No filelist-matched clean audio files found under {clean_root}")

    train_rows: List[dict] = []
    val_rows: List[dict] = []
    generated = 0
    reused = 0
    skipped_clean_files: List[str] = []
    invalid_noise_files: set[Path] = set()
    invalid_rir_files: set[Path] = set()

    for clean_path, clean_key in clean_items:
        ensure_free_space(output_root, free_space_floor_gb, f"synthesize:{source.name}")
        rel_clean = clean_path.relative_to(clean_root)
        speaker_id = lookup_speaker_id(source.name, clean_path, clean_root, speaker_lookup)
        split_key = speaker_id or clean_key or rel_clean.as_posix()
        split = split_name_for(source.name, split_key, val_fraction)
        clean_out, _ = output_pair_paths(output_root, split, source.name, rel_clean, 0)
        noisy_targets = [
            output_pair_paths(output_root, split, source.name, rel_clean, aug_idx)[1]
            for aug_idx in range(max_augmentations_per_clean)
        ]

        if clean_out.exists() and all(noisy_out.exists() for noisy_out in noisy_targets):
            reused += 1 + len(noisy_targets)
            for noisy_out in noisy_targets:
                row = {"noisy": noisy_out.resolve().as_posix(), "clean": clean_out.resolve().as_posix()}
                if split == "train":
                    train_rows.append(row)
                else:
                    val_rows.append(row)
            continue

        try:
            clean_base = peak_normalize(read_audio_full(clean_path, TARGET_SR))
        except Exception as exc:
            skipped_clean_files.append(clean_path.resolve().as_posix())
            log(f"[warn] {source.name}: skipping unreadable clean file {clean_path} ({exc})")
            continue
        clean_base = level_to_dbfs(clean_base, DEFAULT_TARGET_DBFS)
        if clean_base.numel() == 0:
            skipped_clean_files.append(clean_path.resolve().as_posix())
            log(f"[warn] {source.name}: skipping empty clean file {clean_path}")
            continue

        ensure_dir(clean_out.parent)
        if not clean_out.exists():
            torchaudio.save(clean_out.as_posix(), clean_base.unsqueeze(0), TARGET_SR)
        else:
            reused += 1

        for aug_idx in range(max_augmentations_per_clean):
            ensure_free_space(output_root, free_space_floor_gb, f"synthesize:{source.name}:aug{aug_idx + 1}")
            _, noisy_out = output_pair_paths(output_root, split, source.name, rel_clean, aug_idx)
            ensure_dir(noisy_out.parent)
            if not noisy_out.exists():
                rng = random.Random(stable_seed(f"{source.name}:{rel_clean.as_posix()}:{aug_idx}"))
                dry_clean = clean_base
                wet_clean = dry_clean
                if rir_files and rng.random() < rir_probability:
                    rir = sample_valid_audio_full(
                        rir_files,
                        target_sr=TARGET_SR,
                        rng=rng,
                        bad_paths=invalid_rir_files,
                        source_name=source.name,
                        kind="rir",
                    )
                    if rir is not None:
                        wet_clean = apply_rir(dry_clean, rir)
                noise = sample_valid_audio_window(
                    noise_files,
                    target_sr=TARGET_SR,
                    target_frames=wet_clean.numel(),
                    rng=rng,
                    bad_paths=invalid_noise_files,
                    source_name=source.name,
                    kind="noise",
                )
                noisy = mix_clean_and_noise(wet_clean, noise, rng, snr_lower, snr_upper)
                noisy = level_to_dbfs(noisy, DEFAULT_TARGET_DBFS)
                torchaudio.save(noisy_out.as_posix(), noisy.unsqueeze(0), TARGET_SR)
                generated += 1
            else:
                reused += 1

            row = {"noisy": noisy_out.resolve().as_posix(), "clean": clean_out.resolve().as_posix()}
            if split == "train":
                train_rows.append(row)
            else:
                val_rows.append(row)

    shard_root = shard_dir(output_root)
    write_manifest(shard_root / f"{source.name}.train.csv", sorted(train_rows, key=lambda row: row["noisy"]))
    write_manifest(shard_root / f"{source.name}.val.csv", sorted(val_rows, key=lambda row: row["noisy"]))
    combine_manifests(output_root)
    write_state(
        output_root,
        source.name,
        "synthesize",
        {
            "discovered_clean_files": len(discovered_clean_files),
            "clean_files": len(clean_items),
            "salvage_fallback": allow_salvage_fallback,
            "generated_pairs": generated,
            "reused_outputs": reused,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "skipped_clean_files": len(skipped_clean_files),
            "skipped_unlisted_clean_files": len(skipped_unlisted_clean_files),
            "duplicate_clean_files": len(duplicate_clean_files),
            "invalid_noise_files": len(invalid_noise_files),
            "invalid_rir_files": len(invalid_rir_files),
            "skipped_clean_samples": skipped_clean_files[:20],
            "skipped_unlisted_clean_samples": skipped_unlisted_clean_files[:20],
            "duplicate_clean_samples": duplicate_clean_files[:20],
            "invalid_noise_samples": [
                path.resolve().as_posix()
                for path in sorted(invalid_noise_files, key=lambda p: p.as_posix())[:20]
            ],
            "invalid_rir_samples": [
                path.resolve().as_posix()
                for path in sorted(invalid_rir_files, key=lambda p: p.as_posix())[:20]
            ],
        },
    )


def verify_manifest_rows(rows: Sequence[dict]) -> tuple[int, int]:
    checked = 0
    frames_delta = 0
    for row in rows:
        noisy = Path(row["noisy"])
        clean = Path(row["clean"])
        if not noisy.exists():
            raise FileNotFoundError(noisy)
        if not clean.exists():
            raise FileNotFoundError(clean)
        noisy_info = sf.info(noisy)
        clean_info = sf.info(clean)
        if noisy_info.samplerate != TARGET_SR or clean_info.samplerate != TARGET_SR:
            raise ValueError(f"Expected 16kHz for pair: {noisy} / {clean}")
        if noisy_info.channels != 1 or clean_info.channels != 1:
            raise ValueError(f"Expected mono pair: {noisy} / {clean}")
        frames_delta += abs(noisy_info.frames - clean_info.frames)
        if abs(noisy_info.frames - clean_info.frames) > 2:
            raise ValueError(f"Length mismatch: {noisy} vs {clean}")
        checked += 1
    return checked, frames_delta


def verify_sources(
    sources: Sequence[SourceSpec],
    output_root: Path,
) -> None:
    for source in sources:
        if source.kind != "clean":
            continue
        shard_root = shard_dir(output_root)
        rows_train: List[dict] = []
        rows_val: List[dict] = []
        for split, bucket in (("train", rows_train), ("val", rows_val)):
            csv_path = shard_root / f"{source.name}.{split}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                bucket.extend(csv.DictReader(f))
        checked_train, delta_train = verify_manifest_rows(rows_train)
        checked_val, delta_val = verify_manifest_rows(rows_val)
        write_state(
            output_root,
            source.name,
            "verify",
            {
                "checked_train": checked_train,
                "checked_val": checked_val,
                "frames_delta_train": delta_train,
                "frames_delta_val": delta_val,
            },
        )


def all_clean_verified(output_root: Path) -> bool:
    return all(read_state(output_root, name, "verify") is not None for name in CLEAN_SOURCE_ORDER)


def cleanup_trash_root(staging_root: Path) -> Path:
    return ensure_dir(staging_root / ".cleanup_trash")


def cleanup_trash_logs_dir(staging_root: Path) -> Path:
    return ensure_dir(cleanup_trash_root(staging_root) / "logs")


def extract_failure_root(staging_root: Path) -> Path:
    return ensure_dir(staging_root / ".extract_failures")


def quarantine_paths(
    *,
    staging_root: Path,
    source_name: str,
    paths: Sequence[Path],
) -> tuple[Path | None, List[Path]]:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return None, []
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_root = ensure_dir(cleanup_trash_root(staging_root) / f"{source_name}-{batch_id}")
    quarantined: List[Path] = []
    staging_resolved = staging_root.resolve()
    for path in existing_paths:
        try:
            rel = path.resolve().relative_to(staging_resolved)
        except Exception:
            rel = Path(path.name)
        dest = batch_root / rel
        ensure_dir(dest.parent)
        path.rename(dest)
        quarantined.append(dest)
    return batch_root, quarantined


def stash_paths(
    *,
    target_root: Path,
    source_name: str,
    paths: Sequence[Path],
) -> tuple[Path | None, List[Path]]:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return None, []
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_root = ensure_dir(target_root / f"{source_name}-{batch_id}")
    stashed: List[Path] = []
    for path in existing_paths:
        dest = batch_root / path.name
        if path.is_dir():
            path.rename(dest)
        else:
            path.rename(dest)
        stashed.append(dest)
    return batch_root, stashed


def spawn_quarantine_purge(
    *,
    staging_root: Path,
    source_name: str,
    quarantined_paths: Sequence[Path],
) -> tuple[int | None, Path | None]:
    if not quarantined_paths:
        return None, None
    log_path = cleanup_trash_logs_dir(staging_root) / f"{source_name}.purge.log"
    cmd = [
        sys.executable,
        __file__,
        "purge-trash",
        "--trash-path",
        *[str(path) for path in quarantined_paths],
    ]
    with log_path.open("ab") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=lambda: os.nice(19),
        )
    return proc.pid, log_path


def load_source_filelist_entries(source: SourceSpec, staging_root: Path) -> Dict[str, str]:
    root = source_filelist_root(staging_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Missing extracted filelists at {root}. Run extract --source filelists_headset first."
        )
    entries: Dict[str, str] = {}
    collisions: set[str] = set()
    for csv_name in source.filelists:
        csv_path = root / csv_name
        if not csv_path.exists():
            continue
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_name = row.get("filename", "")
                speaker_id = row.get("speaker_id", "").strip()
                canonical_key = canonical_source_key(source.name, raw_name)
                if not canonical_key:
                    continue
                if canonical_key in collisions:
                    continue
                existing = entries.get(canonical_key)
                if existing is None or existing == speaker_id:
                    entries[canonical_key] = speaker_id
                else:
                    collisions.add(canonical_key)
                    entries.pop(canonical_key, None)
    return entries


def source_output_root(output_root: Path, split: str, source_name: str, kind: str) -> Path:
    return output_root / f"{kind}_{split}" / source_name


def discover_output_rows_from_manifests(
    *,
    source: SourceSpec,
    output_root: Path,
) -> tuple[Dict[str, Dict[str, dict]], Dict[str, int]]:
    by_split: Dict[str, Dict[str, dict]] = {"train": {}, "val": {}}
    stats = {
        "manifest_train_input_rows": 0,
        "manifest_val_input_rows": 0,
        "manifest_duplicate_clean_keys": 0,
    }
    shard_root = shard_dir(output_root)
    for split in ("train", "val"):
        csv_path = shard_root / f"{source.name}.{split}.csv"
        if not csv_path.exists():
            continue
        clean_root = source_output_root(output_root, split, source.name, "clean")
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats[f"manifest_{split}_input_rows"] += 1
                clean_path = Path(row["clean"])
                try:
                    rel_clean = clean_path.relative_to(clean_root)
                except ValueError:
                    continue
                canonical_key = canonical_source_key(source.name, rel_clean.as_posix())
                entry = by_split[split].setdefault(
                    canonical_key,
                    {
                        "clean": row["clean"],
                        "noisy": [],
                    },
                )
                if entry["clean"] != row["clean"]:
                    stats["manifest_duplicate_clean_keys"] += 1
                entry["noisy"].append(row["noisy"])
        for entry in by_split[split].values():
            entry["noisy"].sort()
    return by_split, stats


def discover_existing_output_pairs(
    *,
    source: SourceSpec,
    output_root: Path,
) -> tuple[Dict[str, Dict[str, dict]], Dict[str, int]]:
    by_split: Dict[str, Dict[str, dict]] = {"train": {}, "val": {}}
    stats = {
        "clean_train_files": 0,
        "clean_val_files": 0,
        "noisy_train_files": 0,
        "noisy_val_files": 0,
        "duplicate_clean_keys": 0,
    }
    for split in ("train", "val"):
        clean_root = source_output_root(output_root, split, source.name, "clean")
        noisy_root = source_output_root(output_root, split, source.name, "noisy")
        noisy_map: Dict[str, List[str]] = {}
        for noisy_path in audio_files(noisy_root):
            stats[f"noisy_{split}_files"] += 1
            rel = noisy_path.relative_to(noisy_root).as_posix()
            if rel.lower().endswith(".wav"):
                rel = rel[:-4]
            base_name, _, aug_suffix = rel.rpartition("__aug")
            rel_clean = Path(base_name if aug_suffix else rel).with_suffix(".wav")
            canonical_key = canonical_source_key(source.name, rel_clean.as_posix())
            noisy_map.setdefault(canonical_key, []).append(noisy_path.resolve().as_posix())
        for clean_path in audio_files(clean_root):
            stats[f"clean_{split}_files"] += 1
            rel_clean = clean_path.relative_to(clean_root)
            canonical_key = canonical_source_key(source.name, rel_clean.as_posix())
            existing = by_split[split].get(canonical_key)
            clean_abs = clean_path.resolve().as_posix()
            if existing is not None and existing["clean"] != clean_abs:
                stats["duplicate_clean_keys"] += 1
                continue
            by_split[split][canonical_key] = {
                "clean": clean_abs,
                "noisy": sorted(noisy_map.get(canonical_key, [])),
            }
        for canonical_key, noisy_paths in noisy_map.items():
            by_split[split].setdefault(canonical_key, {"clean": None, "noisy": sorted(noisy_paths)})
    return by_split, stats


def rebuild_manifests_from_outputs(
    *,
    source: SourceSpec,
    staging_root: Path,
    output_root: Path,
    val_fraction: float,
) -> dict:
    if source.kind != "clean":
        raise ValueError(f"rebuild-manifests expects clean sources, got {source.name}:{source.kind}")
    official_entries = load_source_filelist_entries(source, staging_root)
    if not official_entries:
        raise RuntimeError(f"No filelist entries found for {source.name}")
    output_pairs, output_stats = discover_output_rows_from_manifests(source=source, output_root=output_root)
    if not output_pairs["train"] and not output_pairs["val"]:
        output_pairs, output_stats = discover_existing_output_pairs(source=source, output_root=output_root)
    train_rows: List[dict] = []
    val_rows: List[dict] = []
    missing_keys: List[str] = []
    wrong_split_keys: List[str] = []
    used_keys: set[str] = set()
    for canonical_key, speaker_id in sorted(official_entries.items()):
        split_key = speaker_id or canonical_key
        split = split_name_for(source.name, split_key, val_fraction)
        entry = output_pairs[split].get(canonical_key)
        if entry is None or not entry.get("clean") or not entry.get("noisy"):
            other_split = "val" if split == "train" else "train"
            other_entry = output_pairs[other_split].get(canonical_key)
            if other_entry is not None and other_entry.get("clean") and other_entry.get("noisy"):
                wrong_split_keys.append(canonical_key)
            else:
                missing_keys.append(canonical_key)
            continue
        used_keys.add(canonical_key)
        bucket = train_rows if split == "train" else val_rows
        for noisy in sorted(entry["noisy"]):
            bucket.append({"noisy": noisy, "clean": entry["clean"]})
    shard_root = shard_dir(output_root)
    write_manifest(shard_root / f"{source.name}.train.csv", sorted(train_rows, key=lambda row: row["noisy"]))
    write_manifest(shard_root / f"{source.name}.val.csv", sorted(val_rows, key=lambda row: row["noisy"]))
    combine_manifests(output_root)
    extra_train_keys = sorted(set(output_pairs["train"]) - set(official_entries))
    extra_val_keys = sorted(set(output_pairs["val"]) - set(official_entries))
    payload = {
        "official_clean_keys": len(official_entries),
        "manifest_train_rows": len(train_rows),
        "manifest_val_rows": len(val_rows),
        "missing_clean_keys": len(missing_keys),
        "wrong_split_clean_keys": len(wrong_split_keys),
        "extra_train_clean_keys": len(extra_train_keys),
        "extra_val_clean_keys": len(extra_val_keys),
        "used_clean_keys": len(used_keys),
        "missing_clean_key_samples": missing_keys[:20],
        "wrong_split_clean_key_samples": wrong_split_keys[:20],
        "extra_train_clean_key_samples": extra_train_keys[:20],
        "extra_val_clean_key_samples": extra_val_keys[:20],
        **output_stats,
    }
    write_state(output_root, source.name, "manifest", payload)
    return payload


def cleanup_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=False)
    elif path.exists():
        path.unlink()


def cleanup_sources(
    sources: Sequence[SourceSpec],
    staging_root: Path,
    output_root: Path,
    keep_devtest: bool,
) -> None:
    verified_all = all_clean_verified(output_root)
    for source in sources:
        if source.kind == "clean":
            if read_state(output_root, source.name, "verify") is None:
                raise RuntimeError(f"Refusing cleanup for {source.name}: verify marker missing")
            cleanup_targets = [clean_extract_dir(staging_root, source.name)]
            cleanup_targets.extend(archive_path_for_blob(staging_root, blob) for blob in source.blobs)
            batch_root, quarantined = quarantine_paths(
                staging_root=staging_root,
                source_name=source.name,
                paths=cleanup_targets,
            )
            purge_pid, purge_log = spawn_quarantine_purge(
                staging_root=staging_root,
                source_name=source.name,
                quarantined_paths=quarantined,
            )
            write_state(
                output_root,
                source.name,
                "cleanup",
                {
                    "cleanup": "done",
                    "quarantine_batch": str(batch_root) if batch_root else None,
                    "quarantined_paths": [str(path) for path in quarantined],
                    "purge_pid": purge_pid,
                    "purge_log": str(purge_log) if purge_log else None,
                },
            )
        elif source.name == "noise_ir":
            if not verified_all:
                raise RuntimeError("Refusing cleanup for noise_ir: not all clean sources are verified")
            shared = shared_extract_dir(staging_root)
            cleanup_targets: List[Path] = []
            for child_name in ("noise_fullband", "impulse_responses"):
                for path in list(shared.rglob(child_name)):
                    if path.is_dir():
                        cleanup_targets.append(path)
            cleanup_targets.extend(archive_path_for_blob(staging_root, blob) for blob in source.blobs)
            batch_root, quarantined = quarantine_paths(
                staging_root=staging_root,
                source_name=source.name,
                paths=cleanup_targets,
            )
            purge_pid, purge_log = spawn_quarantine_purge(
                staging_root=staging_root,
                source_name=source.name,
                quarantined_paths=quarantined,
            )
            write_state(
                output_root,
                source.name,
                "cleanup",
                {
                    "cleanup": "done",
                    "quarantine_batch": str(batch_root) if batch_root else None,
                    "quarantined_paths": [str(path) for path in quarantined],
                    "purge_pid": purge_pid,
                    "purge_log": str(purge_log) if purge_log else None,
                },
            )
        elif source.name == "filelists_headset":
            if not verified_all:
                raise RuntimeError("Refusing cleanup for filelists_headset: not all clean sources are verified")
            cleanup_targets = [metadata_archive_path(staging_root, source), filelists_extract_dir(staging_root)]
            batch_root, quarantined = quarantine_paths(
                staging_root=staging_root,
                source_name=source.name,
                paths=cleanup_targets,
            )
            purge_pid, purge_log = spawn_quarantine_purge(
                staging_root=staging_root,
                source_name=source.name,
                quarantined_paths=quarantined,
            )
            write_state(
                output_root,
                source.name,
                "cleanup",
                {
                    "cleanup": "done",
                    "quarantine_batch": str(batch_root) if batch_root else None,
                    "quarantined_paths": [str(path) for path in quarantined],
                    "purge_pid": purge_pid,
                    "purge_log": str(purge_log) if purge_log else None,
                },
            )
        elif source.name == "devtest":
            if keep_devtest:
                write_state(output_root, source.name, "cleanup", {"cleanup": "kept"})
                continue
            cleanup_targets = [metadata_archive_path(staging_root, source), devtest_extract_dir(output_root)]
            batch_root, quarantined = quarantine_paths(
                staging_root=staging_root,
                source_name=source.name,
                paths=cleanup_targets,
            )
            purge_pid, purge_log = spawn_quarantine_purge(
                staging_root=staging_root,
                source_name=source.name,
                quarantined_paths=quarantined,
            )
            write_state(
                output_root,
                source.name,
                "cleanup",
                {
                    "cleanup": "done",
                    "quarantine_batch": str(batch_root) if batch_root else None,
                    "quarantined_paths": [str(path) for path in quarantined],
                    "purge_pid": purge_pid,
                    "purge_log": str(purge_log) if purge_log else None,
                },
            )


def purge_trash(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            log(f"[purge-trash] {path}")
            cleanup_path(path)


def recover_clean_source_after_extract_failure(
    *,
    source: SourceSpec,
    staging_root: Path,
    output_root: Path,
    reason: str,
) -> tuple[Path | None, List[Path]]:
    paths = [clean_extract_dir(staging_root, source.name)]
    paths.extend(archive_path_for_blob(staging_root, blob) for blob in source.blobs)
    batch_root, stashed = stash_paths(
        target_root=extract_failure_root(staging_root),
        source_name=f"{source.name}-extract-failed",
        paths=paths,
    )
    write_state(
        output_root,
        source.name,
        "extract_recovery",
        {
            "reason": reason,
            "stash_batch": str(batch_root) if batch_root else None,
            "stashed_paths": [str(path) for path in stashed],
        },
    )
    return batch_root, stashed


def mark_clean_source_salvage_pending(
    *,
    source: SourceSpec,
    output_root: Path,
    reasons: Sequence[str],
    stash_batches: Sequence[Path | None],
    stashed_paths: Sequence[Path],
) -> None:
    blocked_state = state_file(output_root, source.name, "blocked")
    if blocked_state.exists():
        blocked_state.unlink()
    write_state(
        output_root,
        source.name,
        "salvage_pending",
        {
            "reasons": list(reasons),
            "stash_batches": [str(path) for path in stash_batches if path is not None],
            "stashed_paths": [str(path) for path in stashed_paths],
            "notes": (
                "Automatic salvage queued after repeated extract failures. "
                "The pipeline will keep all recoverable raw artifacts and continue "
                "with the remaining sources instead of failing hard."
            ),
        },
    )


def run_pipeline(args: argparse.Namespace) -> None:
    sources = resolve_sources(args.source)
    names = {source.name for source in sources}
    clean_sources = selected_clean_sources(sources)
    includes_all_clean = {source.name for source in clean_sources} == set(CLEAN_SOURCE_ORDER)

    if "filelists_headset" in names:
        log("[run] filelists_headset: download")
        download_sources([SOURCE_SPECS["filelists_headset"]], args.staging_root, args.output_root, args.free_space_floor_gb)
        log("[run] filelists_headset: extract")
        extract_sources([SOURCE_SPECS["filelists_headset"]], args.staging_root, args.output_root, args.free_space_floor_gb)

    if "noise_ir" in names:
        log("[run] noise_ir: download")
        download_sources([SOURCE_SPECS["noise_ir"]], args.staging_root, args.output_root, args.free_space_floor_gb)
        log("[run] noise_ir: extract")
        extract_sources([SOURCE_SPECS["noise_ir"]], args.staging_root, args.output_root, args.free_space_floor_gb)

    for source in clean_sources:
        log(f"[run] {source.name}: download")
        download_sources([source], args.staging_root, args.output_root, args.free_space_floor_gb)
        log(f"[run] {source.name}: extract")
        try:
            extract_sources([source], args.staging_root, args.output_root, args.free_space_floor_gb)
        except subprocess.CalledProcessError as exc:
            batch_root, stashed = recover_clean_source_after_extract_failure(
                source=source,
                staging_root=args.staging_root,
                output_root=args.output_root,
                reason=f"{type(exc).__name__}: {exc}",
            )
            if not stashed:
                raise
            log(
                f"[warn] {source.name}: extract failed; stashed inputs at "
                f"{batch_root if batch_root else 'n/a'} and retrying download+extract once"
            )
            log(f"[run] {source.name}: download (retry-after-extract-failure)")
            download_sources([source], args.staging_root, args.output_root, args.free_space_floor_gb)
            log(f"[run] {source.name}: extract (retry-after-extract-failure)")
            try:
                extract_sources([source], args.staging_root, args.output_root, args.free_space_floor_gb)
            except subprocess.CalledProcessError as retry_exc:
                retry_batch_root, retry_stashed = recover_clean_source_after_extract_failure(
                    source=source,
                    staging_root=args.staging_root,
                    output_root=args.output_root,
                    reason=f"{type(retry_exc).__name__}: {retry_exc}",
                )
                mark_clean_source_salvage_pending(
                    source=source,
                    output_root=args.output_root,
                    reasons=(
                        f"{type(exc).__name__}: {exc}",
                        f"{type(retry_exc).__name__}: {retry_exc}",
                    ),
                    stash_batches=(batch_root, retry_batch_root),
                    stashed_paths=[*stashed, *retry_stashed],
                )
                log(
                    f"[warn] {source.name}: extract failed twice; marked salvage_pending "
                    "and continuing with the remaining sources"
                )
                continue
        log(f"[run] {source.name}: synthesize")
        synthesize_source(
            source,
            staging_root=args.staging_root,
            output_root=args.output_root,
            free_space_floor_gb=args.free_space_floor_gb,
            max_augmentations_per_clean=args.max_augmentations_per_clean,
            val_fraction=args.val_fraction,
            limit_clean_files=args.limit_clean_files,
            snr_lower=args.snr_lower,
            snr_upper=args.snr_upper,
            rir_probability=args.rir_probability,
        )
        log(f"[run] {source.name}: verify")
        verify_sources([source], args.output_root)
        log(f"[run] {source.name}: cleanup")
        cleanup_sources([source], args.staging_root, args.output_root, keep_devtest=True)

    shared_cleanup: List[SourceSpec] = []
    if "noise_ir" in names:
        shared_cleanup.append(SOURCE_SPECS["noise_ir"])
    if "filelists_headset" in names:
        shared_cleanup.append(SOURCE_SPECS["filelists_headset"])
    if shared_cleanup:
        if includes_all_clean:
            log("[run] shared cleanup")
            cleanup_sources(shared_cleanup, args.staging_root, args.output_root, keep_devtest=True)
        else:
            log("[run] shared cleanup skipped: not all clean DNS5 sources were processed")

    if "devtest" in names:
        log("[run] devtest: download")
        download_sources([SOURCE_SPECS["devtest"]], args.staging_root, args.output_root, args.free_space_floor_gb)
        log("[run] devtest: extract")
        extract_sources([SOURCE_SPECS["devtest"]], args.staging_root, args.output_root, args.free_space_floor_gb)
        if not args.keep_devtest:
            log("[run] devtest: cleanup")
            cleanup_sources([SOURCE_SPECS["devtest"]], args.staging_root, args.output_root, keep_devtest=False)


def subparser_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="Source(s) sau alias(e): smoke, all_clean, all_relevant, plus nume concrete DNS5.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Datasetul final 16k și manifestele. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    p.add_argument(
        "--staging-root",
        type=Path,
        default=DEFAULT_STAGING_ROOT,
        help=f"Arhive + extrase temporare. Default: {DEFAULT_STAGING_ROOT}",
    )
    p.add_argument(
        "--free-space-floor-gb",
        type=float,
        default=DEFAULT_FREE_SPACE_FLOOR_GB,
        help="Prag minim de spațiu liber pe staging/output. Default: 50 GB.",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="DNS5 Track1 Headset pipeline pentru ULP-SE-aTENNuate."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_download = sub.add_parser("download", help="Descarcă arhivele DNS5 relevante.")
    subparser_common(p_download)

    p_extract = sub.add_parser("extract", help="Extrage arhivele în staging sau devtest final.")
    subparser_common(p_extract)

    p_synth = sub.add_parser("synthesize", help="Generează perechi 16k noisy/clean offline.")
    subparser_common(p_synth)
    p_synth.add_argument(
        "--max-augmentations-per-clean",
        type=int,
        default=DEFAULT_MAX_AUGS,
        help="Număr maxim de variante noisy per fișier clean. Default: 1.",
    )
    p_synth.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fracția de validare speaker-disjoint. Default: 0.1.",
    )
    p_synth.add_argument(
        "--limit-clean-files",
        type=int,
        default=None,
        help="Limitează numărul de fișiere clean pentru smoke tests.",
    )
    p_synth.add_argument("--snr-lower", type=float, default=DEFAULT_SNR_LOWER)
    p_synth.add_argument("--snr-upper", type=float, default=DEFAULT_SNR_UPPER)
    p_synth.add_argument("--rir-probability", type=float, default=DEFAULT_RIR_PROBABILITY)

    p_verify = sub.add_parser("verify", help="Verifică manifestele și fișierele 16k generate.")
    subparser_common(p_verify)

    p_rebuild = sub.add_parser(
        "rebuild-manifests",
        help="Reconstruiește manifestele strict din filelistul oficial și outputul existent.",
    )
    subparser_common(p_rebuild)
    p_rebuild.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fracția de validare speaker-disjoint. Default: 0.1.",
    )

    p_cleanup = sub.add_parser("cleanup", help="Șterge arhivele/raw doar după verify cu succes.")
    subparser_common(p_cleanup)
    p_cleanup.add_argument(
        "--keep-devtest",
        action="store_true",
        help="Păstrează Track1 dev-test după cleanup.",
    )

    p_purge = sub.add_parser("purge-trash", help=argparse.SUPPRESS)
    p_purge.add_argument(
        "--trash-path",
        nargs="+",
        type=Path,
        required=True,
        help=argparse.SUPPRESS,
    )

    p_run = sub.add_parser("run", help="Rulează pipeline-ul complet pe loturi, end-to-end.")
    subparser_common(p_run)
    p_run.add_argument(
        "--max-augmentations-per-clean",
        type=int,
        default=DEFAULT_MAX_AUGS,
        help="Număr maxim de variante noisy per fișier clean. Default: 1.",
    )
    p_run.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fracția de validare speaker-disjoint. Default: 0.1.",
    )
    p_run.add_argument(
        "--limit-clean-files",
        type=int,
        default=None,
        help="Limitează numărul de fișiere clean pentru smoke tests.",
    )
    p_run.add_argument("--snr-lower", type=float, default=DEFAULT_SNR_LOWER)
    p_run.add_argument("--snr-upper", type=float, default=DEFAULT_SNR_UPPER)
    p_run.add_argument("--rir-probability", type=float, default=DEFAULT_RIR_PROBABILITY)
    p_run.add_argument(
        "--keep-devtest",
        action="store_true",
        help="Păstrează Track1 dev-test după finalizarea pipeline-ului.",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "purge-trash":
        purge_trash(args.trash_path)
        return 0

    ensure_dir(args.output_root)
    ensure_dir(args.staging_root)

    sources = resolve_sources(args.source)
    if args.cmd == "download":
        download_sources(sources, args.staging_root, args.output_root, args.free_space_floor_gb)
        return 0
    if args.cmd == "extract":
        extract_sources(sources, args.staging_root, args.output_root, args.free_space_floor_gb)
        return 0
    if args.cmd == "synthesize":
        clean_sources = [source for source in sources if source.kind == "clean"]
        if not clean_sources:
            raise ValueError("synthesize expects at least one clean source or alias that resolves to clean sources")
        for source in clean_sources:
            synthesize_source(
                source,
                staging_root=args.staging_root,
                output_root=args.output_root,
                free_space_floor_gb=args.free_space_floor_gb,
                max_augmentations_per_clean=args.max_augmentations_per_clean,
                val_fraction=args.val_fraction,
                limit_clean_files=args.limit_clean_files,
                snr_lower=args.snr_lower,
                snr_upper=args.snr_upper,
                rir_probability=args.rir_probability,
            )
        return 0
    if args.cmd == "verify":
        verify_sources(sources, args.output_root)
        return 0
    if args.cmd == "rebuild-manifests":
        clean_sources = [source for source in sources if source.kind == "clean"]
        if not clean_sources:
            raise ValueError(
                "rebuild-manifests expects at least one clean source or alias that resolves to clean sources"
            )
        for source in clean_sources:
            payload = rebuild_manifests_from_outputs(
                source=source,
                staging_root=args.staging_root,
                output_root=args.output_root,
                val_fraction=args.val_fraction,
            )
            if payload["missing_clean_keys"] or payload["wrong_split_clean_keys"]:
                raise RuntimeError(
                    f"Manifest rebuild for {source.name} incomplete: "
                    f"missing={payload['missing_clean_keys']} wrong_split={payload['wrong_split_clean_keys']}"
                )
        return 0
    if args.cmd == "cleanup":
        cleanup_sources(
            sources,
            args.staging_root,
            args.output_root,
            keep_devtest=args.keep_devtest,
        )
        return 0
    if args.cmd == "run":
        run_pipeline(args)
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    sys.exit(main())
