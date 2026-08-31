"""
Aggregates classification benchmark results across seeds into mean ± std.

Each run of `classification_benchmark.py` writes one CSV for a single seed.
This script combines several of them into the table reported in README.md.
"""

import argparse
from pathlib import Path

import pandas as pd

METRICS = {
    "ID Acc": ("ID Acc ↑", 100, "%"),
    "ID ECE": ("ECE ↓", 1, ""),
    "Shift Acc": ("Shift Acc ↑", 100, "%"),
    "OOD AUROC": ("OOD AUROC ↑", 1, ""),
}


def load_runs(paths: list[Path]) -> pd.DataFrame:
    """Reads every per-seed CSV into one frame.

    Args:
        paths: Result directories, each holding a `benchmark_results.csv`.

    Returns:
        pd.DataFrame: Concatenated rows with a `Run` column naming the source.

    Raises:
        FileNotFoundError: If none of the paths holds a results CSV.
    """
    frames = []
    for path in paths:
        csv_path = path / "benchmark_results.csv" if path.is_dir() else path
        if not csv_path.exists():
            print(f"skipping {csv_path}, not found")
            continue
        frame = pd.read_csv(csv_path)
        frame["Run"] = csv_path.parent.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("no benchmark_results.csv found in the given paths")

    print(f"aggregating {len(frames)} runs: {', '.join(f['Run'][0] for f in frames)}")
    return pd.concat(frames, ignore_index=True)


def to_markdown(df: pd.DataFrame) -> str:
    """Formats per-seed results as a markdown table of mean ± std.

    Args:
        df: Concatenated per-seed results.

    Returns:
        str: Markdown table, methods in rows and metrics in columns.
    """
    methods = df["Method"].drop_duplicates().tolist()
    headers = [label for label, _, _ in METRICS.values()]

    lines = [
        "| Method | " + " | ".join(headers) + " |",
        "| ------ |" + " -----------: |" * len(headers),
    ]
    for method in methods:
        rows = df[df["Method"] == method]
        cells = []
        for column, (_, scale, suffix) in METRICS.items():
            values = rows[column] * scale
            digits = 2 if scale == 100 else 4
            cells.append(
                f"{values.mean():.{digits}f} ± {values.std(ddof=0):.{digits}f}{suffix}"
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """Reads per-seed results and prints the aggregated markdown table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="result directories, one per seed (e.g. results/seed_*)",
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="also write the aggregate to this CSV"
    )
    args = parser.parse_args()

    df = load_runs(args.paths)
    print()
    print(to_markdown(df))

    if args.csv:
        summary = df.groupby("Method", sort=False)[list(METRICS)].agg(["mean", "std"])
        summary.to_csv(args.csv)
        print(f"\nwritten to {args.csv}")


if __name__ == "__main__":
    main()
