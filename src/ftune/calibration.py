"""Hardware calibration for ftune.

Runs a short benchmark (or accepts benchmark results) to compute
hardware-specific multipliers that improve estimation accuracy.

The calibration measures actual MFU and memory overhead on the user's
specific GPU + driver + framework combination, then adjusts future
estimates accordingly.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ftune.core.models import CalibrationResult


class Calibrator:
    """Hardware calibration to close the estimation accuracy gap.

    Instead of relying on generic MFU constants, calibration measures
    actual throughput on the user's hardware and computes correction
    multipliers.

    Usage:
        # Option 1: From manual benchmark results
        cal = Calibrator.from_benchmark(
            estimated_memory_gb=9.09,
            actual_memory_gb=11.2,
            estimated_time_hours=32.9,
            actual_time_hours=4.5,
            gpu_name="A100-80GB",
        )

        # Option 2: From a PyTorch benchmark (requires torch)
        cal = Calibrator.from_pytorch_benchmark(
            model_name="meta-llama/Llama-3.1-8B",
            method="qlora",
            gpu_name="A100-80GB",
            steps=10,
        )

        # Save calibration for reuse
        cal_result = cal.result
        Calibrator.save(cal_result, "~/.ftune/calibration.json")

        # Load and apply
        cal_result = Calibrator.load("~/.ftune/calibration.json")
        adjusted_time = estimated_time * cal_result.hardware_multiplier
    """

    def __init__(self, result: CalibrationResult) -> None:
        self.result = result

    @staticmethod
    def from_benchmark(
        estimated_memory_gb: float,
        actual_memory_gb: float,
        estimated_time_hours: float,
        actual_time_hours: float,
        gpu_name: str = "",
        steps_benchmarked: int = 10,
    ) -> "Calibrator":
        """Create calibration from manual benchmark comparison.

        Run your actual training for ~10 steps, measure peak memory
        and throughput, then feed those numbers here.

        Args:
            estimated_memory_gb: What ftune predicted.
            actual_memory_gb: Actual peak VRAM from nvidia-smi.
            estimated_time_hours: What ftune predicted for total training.
            actual_time_hours: Actual training time (or extrapolated from benchmark).
            gpu_name: GPU used for benchmark.
            steps_benchmarked: Number of steps in the benchmark.

        Returns:
            Calibrator with computed hardware multipliers.
        """
        notes = []

        # Memory multiplier: how much to scale memory estimates
        memory_multiplier = actual_memory_gb / estimated_memory_gb if estimated_memory_gb > 0 else 1.0

        # Hardware multiplier: how much to scale time estimates
        # < 1.0 means actual is faster than estimated (good)
        # > 1.0 means actual is slower than estimated
        hardware_multiplier = actual_time_hours / estimated_time_hours if estimated_time_hours > 0 else 1.0

        # Derive actual MFU from the multiplier
        # If estimated MFU was 0.35 and actual was 2x faster, real MFU ≈ 0.7
        base_mfu = 0.35  # Our default
        measured_mfu = base_mfu / hardware_multiplier if hardware_multiplier > 0 else base_mfu

        # Clamp MFU to reasonable range
        measured_mfu = max(0.05, min(0.85, measured_mfu))

        # Compute throughput
        measured_throughput = 0.0  # Would need tokens/sec from benchmark

        if memory_multiplier > 1.3:
            notes.append(
                f"Memory underestimated by {(memory_multiplier - 1) * 100:.0f}%. "
                f"Framework overhead is higher than expected."
            )
        elif memory_multiplier < 0.7:
            notes.append(
                f"Memory overestimated by {(1 - memory_multiplier) * 100:.0f}%. "
                f"Estimates are conservative for this setup."
            )

        if hardware_multiplier > 2.0:
            notes.append(
                f"Training is {hardware_multiplier:.1f}x slower than theoretical. "
                f"This is typical for I/O-bound workloads or unoptimized dataloaders."
            )
        elif hardware_multiplier < 0.5:
            notes.append(
                f"Training is {1/hardware_multiplier:.1f}x faster than theoretical. "
                f"Your setup is well-optimized."
            )

        return Calibrator(
            result=CalibrationResult(
                measured_mfu=round(measured_mfu, 4),
                measured_memory_gb=actual_memory_gb,
                measured_throughput_tokens_per_sec=measured_throughput,
                hardware_multiplier=round(hardware_multiplier, 4),
                memory_multiplier=round(memory_multiplier, 4),
                gpu_name=gpu_name,
                steps_benchmarked=steps_benchmarked,
                notes=notes,
            )
        )

    @staticmethod
    def from_pytorch_log(
        log_data: dict,
        estimated_memory_gb: float,
        estimated_time_hours: float,
    ) -> "Calibrator":
        """Create calibration from PyTorch training log output.

        Parses common training framework log formats (HuggingFace Trainer,
        Axolotl, custom loops) to extract benchmark metrics.

        Args:
            log_data: Dict with keys like 'peak_memory_gb', 'tokens_per_second',
                      'steps_per_second', 'total_steps', 'total_time_seconds'.
            estimated_memory_gb: ftune's memory estimate.
            estimated_time_hours: ftune's time estimate.
        """
        actual_memory = log_data.get("peak_memory_gb", estimated_memory_gb)

        # Compute actual time from various possible keys
        actual_time = None
        if "total_time_hours" in log_data:
            actual_time = log_data["total_time_hours"]
        elif "total_time_seconds" in log_data:
            actual_time = log_data["total_time_seconds"] / 3600
        elif "steps_per_second" in log_data and "total_steps" in log_data:
            total_seconds = log_data["total_steps"] / log_data["steps_per_second"]
            actual_time = total_seconds / 3600

        if actual_time is None:
            actual_time = estimated_time_hours

        return Calibrator.from_benchmark(
            estimated_memory_gb=estimated_memory_gb,
            actual_memory_gb=actual_memory,
            estimated_time_hours=estimated_time_hours,
            actual_time_hours=actual_time,
            gpu_name=log_data.get("gpu_name", ""),
        )

    def adjust_time(self, estimated_hours: float) -> float:
        """Apply calibration to a time estimate.

        Args:
            estimated_hours: Raw ftune time estimate.

        Returns:
            Calibrated time estimate.
        """
        return estimated_hours * self.result.hardware_multiplier

    def adjust_memory(self, estimated_gb: float) -> float:
        """Apply calibration to a memory estimate.

        Args:
            estimated_gb: Raw ftune memory estimate.

        Returns:
            Calibrated memory estimate.
        """
        return estimated_gb * self.result.memory_multiplier

    def format_report(self) -> str:
        """Generate a human-readable calibration report."""
        r = self.result
        lines = [
            "╭────────────────────────────────────────────────╮",
            "│        🔧 ftune Calibration Report              │",
            "╰────────────────────────────────────────────────╯",
            "",
            f"  GPU: {r.gpu_name or 'Unknown'}",
            f"  Steps benchmarked: {r.steps_benchmarked}",
            "",
            f"  Measured MFU:          {r.measured_mfu:.2%}",
            f"  Hardware multiplier:   {r.hardware_multiplier:.2f}x",
            f"  Memory multiplier:     {r.memory_multiplier:.2f}x",
            "",
            "  How to use:",
            f"    Multiply time estimates by {r.hardware_multiplier:.2f}",
            f"    Multiply memory estimates by {r.memory_multiplier:.2f}",
            f"    Or use mfu_override={r.measured_mfu:.4f} in estimate_time()",
            "",
        ]
        if r.notes:
            lines.append("  Notes:")
            for note in r.notes:
                lines.append(f"    • {note}")
        return "\n".join(lines)

    @staticmethod
    def save(result: CalibrationResult, path: str) -> None:
        """Save calibration result to JSON file."""
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(asdict(result), f, indent=2)

    @staticmethod
    def load(path: str) -> CalibrationResult:
        """Load calibration result from JSON file."""
        p = Path(path).expanduser()
        with open(p, "r") as f:
            data = json.load(f)
        return CalibrationResult(**data)
