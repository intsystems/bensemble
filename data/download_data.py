#!/usr/bin/env python3
"""
Download the datasets needed by the benchmark notebooks.

Currently fetches the Boston Housing dataset used in `benchmark_extended.ipynb`
and saves it to the repository-level `data/` directory as `housing.data`.
"""

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

BOSTON_HOUSING_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
)

# Resolve paths relative to this repository, independent of the current workdir
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEST_PATH = DATA_DIR / "housing.data"


def download_file(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, creating parent directories if needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already present, skipping download: {dest}")
        return

    try:
        with urlopen(url) as response:
            data = response.read()
    except URLError as exc:
        raise SystemExit(f"Failed to download {url}: {exc}") from exc

    dest.write_bytes(data)
    print(f"Downloaded {url} -> {dest}")


def main() -> None:
    download_file(BOSTON_HOUSING_URL, DEST_PATH)
    print(f"Boston Housing dataset ready at: {DEST_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
