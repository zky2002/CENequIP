#!/usr/bin/env python
"""Download and verify all revised MD17 datasets used by NequIP."""

from __future__ import annotations

import argparse
from pathlib import Path

from nequip.data.datamodule.rmd17_datamodule import rMD17DataModule
from nequip.utils import extract_tar
from nequip.utils.file_utils import download_url


# Direct archive URL for the rMD17 tarball.
DATASET_URL = "https://archive.materialscloud.org/record/file?filename=rmd17.tar.bz2&record_id=466"
ARCHIVE_NAME = "rmd17.tar.bz2"
DATASET_FILES = rMD17DataModule.DATASET_MAP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source-dir",
        default="data",
        help="Directory containing or receiving the rmd17/npz_data folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_source_dir = Path(args.data_source_dir)
    npz_dir = data_source_dir / "rmd17" / "npz_data"
    missing = [
        file_name
        for file_name in DATASET_FILES.values()
        if not (npz_dir / file_name).is_file()
    ]

    if missing:
        print(f"Missing {len(missing)} rMD17 files; downloading archive to {data_source_dir}")
        archive_path = data_source_dir / ARCHIVE_NAME
        if archive_path.exists() and archive_path.stat().st_size == 0:
            archive_path.unlink()
        download_path = download_url(DATASET_URL, str(data_source_dir), ARCHIVE_NAME)
        extract_tar(
            path=download_path,
            folder=str(data_source_dir),
            mode="r:bz2",
        )
    else:
        print(f"All rMD17 npz files already exist in {npz_dir}")

    print("\nVerified rMD17 files:")
    missing_after = []
    for dataset, file_name in DATASET_FILES.items():
        path = npz_dir / file_name
        if path.is_file():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {dataset:14s} {file_name:28s} {size_mb:8.1f} MB")
        else:
            missing_after.append(str(path))

    if missing_after:
        raise FileNotFoundError(
            "The following rMD17 files are still missing:\n"
            + "\n".join(missing_after)
        )


if __name__ == "__main__":
    main()
