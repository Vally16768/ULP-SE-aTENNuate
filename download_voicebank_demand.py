from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


DATASET_ASSETS = [
    {
        "name": "clean_testset_wav.zip",
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip?sequence=1&isAllowed=y",
        "sentinel": "clean_testset_wav",
    },
    {
        "name": "clean_trainset_28spk_wav.zip",
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?sequence=2&isAllowed=y",
        "sentinel": "clean_trainset_28spk_wav",
    },
    {
        "name": "noisy_testset_wav.zip",
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip?sequence=5&isAllowed=y",
        "sentinel": "noisy_testset_wav",
    },
    {
        "name": "noisy_trainset_28spk_wav.zip",
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip?sequence=6&isAllowed=y",
        "sentinel": "noisy_trainset_28spk_wav",
    },
]


def _download_with_progress(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    tmp_path = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", "0"))
        with tmp_path.open("wb") as handle, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"download:{dst.name}",
        ) as progress:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))
    shutil.move(tmp_path.as_posix(), dst.as_posix())
    return {"path": dst.as_posix(), "bytes": int(dst.stat().st_size)}


def _extract_zip(archive: Path, out_root: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_root)
        members = zf.namelist()
    return {"archive": archive.as_posix(), "members": len(members)}


def ensure_voicebank_raw_dataset(
    raw_root: str | Path,
    *,
    include_metadata: bool = True,
) -> dict[str, Any]:
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    downloads_dir = raw_root / "_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    downloads: list[dict[str, Any]] = []
    extracts: list[dict[str, Any]] = []

    for asset in DATASET_ASSETS:
        sentinel = raw_root / asset["sentinel"]
        archive_path = downloads_dir / asset["name"]
        if sentinel.exists():
            downloads.append({"name": asset["name"], "status": "already_present", "path": archive_path.as_posix()})
            continue
        if not archive_path.exists():
            info = _download_with_progress(asset["url"], archive_path)
            downloads.append({"name": asset["name"], "status": "downloaded", **info})
        else:
            downloads.append(
                {
                    "name": asset["name"],
                    "status": "archive_present",
                    "path": archive_path.as_posix(),
                    "bytes": int(archive_path.stat().st_size),
                }
            )
        extracts.append({"name": asset["name"], **_extract_zip(archive_path, raw_root)})

    summary = {
        "raw_root": raw_root.as_posix(),
        "downloads": downloads,
        "extracts": extracts,
        "source": {
            "provider": "University of Edinburgh DataShare",
            "landing_page": "https://datashare.ed.ac.uk/handle/10283/2791",
            "doi": "10.7488/ds/2117",
        },
    }
    if include_metadata:
        (raw_root / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official VoiceBank-DEMAND dataset into the raw root.")
    parser.add_argument("--out-root", type=str, default="dataset/voicebank-demand/raw")
    args = parser.parse_args()
    summary = ensure_voicebank_raw_dataset(args.out_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
