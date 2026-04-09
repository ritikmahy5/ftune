"""Summarize benchmark results into a validation table for the README."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_results(results_dir: str) -> list[dict]:
    """Load all JSON result files from a directory."""
    results = []
    for f in Path(results_dir).glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
    return results


def print_markdown_table(results: list[dict]) -> None:
    """Print a markdown validation table."""
    print("\n## Validation Results\n")
    print("| Model | Method | GPU | ftune VRAM | Actual VRAM | Mem Error | "
          "ftune Time | Actual Time | Time Error |")
    print("|---|---|---|---|---|---|---|---|---|")

    for r in results:
        c = r.get("comparison", {})
        if not c:
            continue

        mem_err = c["memory_error_pct"]
        time_err = c["time_error_pct"]
        mem_grade = "within 20%" if abs(mem_err) < 20 else "needs calibration"
        time_grade = "within 2x" if abs(time_err) < 100 else "needs calibration"

        quant = f" ({r['quantization']})" if r["quantization"] != "none" else ""

        print(f"| {r['model'].split('/')[-1]} | {r['method']}{quant} | "
              f"{r['gpu_name']} | "
              f"{c['ftune_memory_gb']:.1f} GB | {c['actual_memory_gb']:.1f} GB | "
              f"{mem_err:+.1f}% | "
              f"{c['ftune_time_hours']:.1f}h | {c['actual_time_hours_extrapolated']:.1f}h | "
              f"{time_err:+.1f}% |")

    print()


def print_summary(results: list[dict]) -> None:
    """Print summary statistics."""
    mem_errors = []
    time_errors = []

    for r in results:
        c = r.get("comparison", {})
        if c:
            mem_errors.append(abs(c["memory_error_pct"]))
            time_errors.append(abs(c["time_error_pct"]))

    if not mem_errors:
        print("No comparison data found.")
        return

    print(f"**Summary across {len(mem_errors)} benchmarks:**")
    print(f"- Memory: median error {sorted(mem_errors)[len(mem_errors)//2]:.1f}%, "
          f"max {max(mem_errors):.1f}%")
    print(f"- Time: median error {sorted(time_errors)[len(time_errors)//2]:.1f}%, "
          f"max {max(time_errors):.1f}%")

    within_20 = sum(1 for e in mem_errors if e < 20)
    print(f"- Memory within 20%: {within_20}/{len(mem_errors)}")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}/")
        sys.exit(1)

    print_markdown_table(results)
    print_summary(results)
