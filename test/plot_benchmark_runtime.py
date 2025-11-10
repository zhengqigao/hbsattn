#!/usr/bin/env python3
"""
Utility to visualize runtimes from benchmark_results.json.

The benchmark file is expected to contain one JSON object per benchmark
configuration (for example, in NDJSON format). Each object should include
`unit_seqlen`, plus one or more result entries whose keys map to dictionaries
containing runtime statistics (for example `runtime_mean`, `runtime_std`, etc.).
Scalar metadata fields such as `unit_seqlen`, `headdim`, and `nhead_q` are
ignored for plotting.

Example usage:

    python plot_benchmark_runtime.py \
        --benchmark-file benchmark_results.json \
        --metric runtime_mean \
        --output runtime_vs_unit_seqlen.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_records(path: pathlib.Path) -> List[Dict]:
    """
    Load a benchmark file that stores multiple JSON objects back-to-back.

    This supports files produced by repeatedly dumping JSON without wrapping
    them in a list (i.e. NDJSON-like structure).
    """
    text = path.read_text().strip()
    decoder = json.JSONDecoder()
    records = []
    idx = 0

    while idx < len(text):
        # Skip whitespace between objects
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break

        record, end_idx = decoder.raw_decode(text, idx)
        records.append(record)
        idx = end_idx

    if not records:
        raise ValueError(f"No JSON objects found in {path}")

    return records


def collect_series_with_std(
    records: Iterable[Dict], metric: str
) -> Dict[str, List[Tuple[int, float, float]]]:
    """
    Arrange benchmark records into series keyed by method name,
    collecting (unit_seqlen, value, std) per point.
    Returns a mapping of method -> list of (unit_seqlen, value, std).
    """
    metric_std = metric.replace("mean", "std") if "mean" in metric else metric + "_std"
    series: Dict[str, List[Tuple[int, float, float]]] = {}

    for record in records:
        unit_seqlen = record.get("unit_seqlen")
        if unit_seqlen is None:
            raise KeyError("Each record must include a 'unit_seqlen' field.")
        for key, value in record.items():
            if not isinstance(value, dict):
                continue
            metric_value = value.get(metric, None)
            metric_std_value = value.get(metric_std, None)
            series.setdefault(key, []).append((unit_seqlen, metric_value, metric_std_value))

    # Sort each series by unit_seqlen for nicer plotting
    for data in series.values():
        data.sort(key=lambda pair: pair[0])

    if not series:
        raise ValueError(
            f"Did not find any result entries containing metric '{metric}'."
        )

    return series


def plot_series(
    series: Dict[str, List[Tuple[int, float, float]]],
    metric: str,
    output_path: pathlib.Path | None,
    show: bool,
) -> None:
    """
    Plot each method's runtime metric against unit sequence length.
    Uses log2 scale for x-axis and log scale for y-axis.
    Adds a shaded region to indicate ±1 standard deviation.
    """
    fig, ax = plt.subplots()

    for method, data in sorted(series.items()):
        unit_seqlen, values, stds = zip(*data)
        unit_seqlen_arr = np.array(unit_seqlen)
        # Plot the mean curve
        # Plot the std shaded region
        # To handle possible None values (meaning missing metrics) gracefully, we mask out such points for shading.
        values_valid = np.array([v if v is not None else np.nan for v in values])
        stds_valid = np.array([s if (v is not None and s is not None) else np.nan for v, s in zip(values, stds)])
        # Only plot fill_between for points that have numeric values
        not_nan = ~np.isnan(values_valid) & ~np.isnan(stds_valid)
        ax.plot(unit_seqlen_arr[not_nan], values_valid[not_nan], marker="o", label=method)
        if np.any(not_nan):
            ax.fill_between(
                unit_seqlen_arr[not_nan],
                (values_valid[not_nan] - 3 * stds_valid[not_nan]),
                (values_valid[not_nan] + 3 * stds_valid[not_nan]),
                alpha=0.18
            )

    ax.set_xlabel("unit_seqlen (log2 scale)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs unit_seqlen (log2-x, log-y)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved plot to {output_path}")

    if show or not output_path:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtime metrics vs unit sequence length from benchmark results."
    )
    parser.add_argument(
        "--benchmark-file",
        type=pathlib.Path,
        default=pathlib.Path("benchmark_results.json"),
        help="Path to the benchmark results JSON file.",
    )
    parser.add_argument(
        "--metric",
        default="runtime_mean",
        help="Metric inside each method entry to plot (default: runtime_mean).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional path to save the plot image. If omitted, the plot is shown instead.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Always display the plot window (in addition to saving if --output is provided).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = load_benchmark_records(args.benchmark_file)
    series = collect_series_with_std(records, args.metric)
    plot_series(series, args.metric, args.output, args.show)


if __name__ == "__main__":
    main()

