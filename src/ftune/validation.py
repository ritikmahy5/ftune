"""Validation module — compare ftune estimates against actual training metrics.

Supports importing actual training data from:
  - Manual input (dict)
  - W&B (Weights & Biases) run logs
  - TensorBoard event files
  - JSON/CSV log files

Generates an accuracy report showing how close estimates were to reality.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ActualMetrics:
    """Actual training metrics from a real training run."""

    peak_memory_gb: Optional[float] = None
    training_time_hours: Optional[float] = None
    total_cost: Optional[float] = None
    gpu_name: Optional[str] = None
    num_gpus: int = 1
    dataset_size: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    seq_length: Optional[int] = None
    method: Optional[str] = None
    model_name: Optional[str] = None
    source: str = "manual"  # "manual", "wandb", "tensorboard", "json"
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Comparison between estimated and actual metrics."""

    # Memory
    estimated_memory_gb: Optional[float] = None
    actual_memory_gb: Optional[float] = None
    memory_error_pct: Optional[float] = None
    memory_error_gb: Optional[float] = None

    # Time
    estimated_time_hours: Optional[float] = None
    actual_time_hours: Optional[float] = None
    time_error_pct: Optional[float] = None
    time_error_hours: Optional[float] = None

    # Cost
    estimated_cost: Optional[float] = None
    actual_cost: Optional[float] = None
    cost_error_pct: Optional[float] = None

    # Overall
    accuracy_grade: str = ""  # "Excellent", "Good", "Fair", "Poor"
    notes: List[str] = field(default_factory=list)


def _compute_error(estimated: Optional[float], actual: Optional[float]) -> tuple:
    """Compute absolute and percentage error.

    Returns:
        Tuple of (absolute_error, percentage_error) or (None, None).
    """
    if estimated is None or actual is None or actual == 0:
        return None, None
    error = estimated - actual
    pct = (error / actual) * 100
    return round(error, 2), round(pct, 1)


def _grade_accuracy(errors: List[Optional[float]]) -> str:
    """Grade overall accuracy based on percentage errors."""
    valid_errors = [abs(e) for e in errors if e is not None]
    if not valid_errors:
        return "N/A"
    avg_error = sum(valid_errors) / len(valid_errors)
    if avg_error <= 10:
        return "Excellent (<10% avg error)"
    elif avg_error <= 20:
        return "Good (10-20% avg error)"
    elif avg_error <= 35:
        return "Fair (20-35% avg error)"
    return "Poor (>35% avg error)"


class Validator:
    """Compare ftune estimates against actual training metrics.

    Example:
        >>> from ftune import Estimator
        >>> from ftune.validation import Validator, ActualMetrics
        >>>
        >>> est = Estimator(model="meta-llama/Llama-3.1-8B", method="qlora", quantization="4bit")
        >>> actual = ActualMetrics(peak_memory_gb=11.2, training_time_hours=4.5)
        >>> result = Validator.compare(est, actual)
        >>> print(Validator.format_report(result))
    """

    @staticmethod
    def compare(
        estimator: Any,
        actual: ActualMetrics,
        dataset_size: Optional[int] = None,
        epochs: Optional[int] = None,
    ) -> ValidationResult:
        """Compare estimates against actual metrics.

        Args:
            estimator: ftune Estimator instance.
            actual: Actual training metrics.
            dataset_size: Dataset size (overrides actual.dataset_size if provided).
            epochs: Number of epochs (overrides actual.epochs if provided).

        Returns:
            ValidationResult with error analysis.
        """
        result = ValidationResult()
        notes = []

        # ── Memory comparison ──
        mem = estimator.estimate_memory()
        result.estimated_memory_gb = mem.total_gb

        if actual.peak_memory_gb is not None:
            result.actual_memory_gb = actual.peak_memory_gb
            result.memory_error_gb, result.memory_error_pct = _compute_error(
                mem.total_gb, actual.peak_memory_gb
            )
            if result.memory_error_pct is not None:
                if result.memory_error_pct > 0:
                    notes.append(
                        f"Memory overestimated by {result.memory_error_pct:.1f}% "
                        f"({result.memory_error_gb:+.2f} GB)"
                    )
                else:
                    notes.append(
                        f"Memory underestimated by {abs(result.memory_error_pct):.1f}% "
                        f"({result.memory_error_gb:+.2f} GB) — risk of OOM!"
                    )

        # ── Time comparison ──
        ds = dataset_size or actual.dataset_size
        ep = epochs or actual.epochs
        gpu = actual.gpu_name

        if ds and ep and gpu:
            try:
                time_est = estimator.estimate_time(
                    gpu=gpu, dataset_size=ds, epochs=ep,
                    num_gpus=actual.num_gpus,
                )
                result.estimated_time_hours = time_est.total_hours

                if actual.training_time_hours is not None:
                    result.actual_time_hours = actual.training_time_hours
                    result.time_error_hours, result.time_error_pct = _compute_error(
                        time_est.total_hours, actual.training_time_hours
                    )
            except Exception as e:
                notes.append(f"Could not estimate time: {e}")

        # ── Cost comparison ──
        if actual.total_cost is not None and result.estimated_time_hours:
            # We can't perfectly estimate cost without knowing the provider,
            # but we can compare if the user provides actual cost
            result.actual_cost = actual.total_cost
            if result.actual_time_hours and result.actual_cost > 0:
                actual_rate = result.actual_cost / result.actual_time_hours
                result.estimated_cost = round(actual_rate * result.estimated_time_hours, 2)
                _, result.cost_error_pct = _compute_error(
                    result.estimated_cost, result.actual_cost
                )

        # ── Overall grade ──
        result.accuracy_grade = _grade_accuracy([
            result.memory_error_pct,
            result.time_error_pct,
        ])
        result.notes = notes

        return result

    @staticmethod
    def format_report(result: ValidationResult) -> str:
        """Format a human-readable validation report.

        Args:
            result: ValidationResult from compare().

        Returns:
            Formatted string report.
        """
        lines = [
            "╭────────────────────────────────────────────────╮",
            "│        📊 ftune Validation Report               │",
            "╰────────────────────────────────────────────────╯",
            "",
        ]

        # Memory
        if result.actual_memory_gb is not None:
            icon = "✅" if abs(result.memory_error_pct or 0) <= 20 else "⚠️"
            lines.extend([
                "Memory:",
                f"  Estimated:  {result.estimated_memory_gb:.2f} GB",
                f"  Actual:     {result.actual_memory_gb:.2f} GB",
                f"  {icon} Error:  {result.memory_error_gb:+.2f} GB ({result.memory_error_pct:+.1f}%)",
                "",
            ])

        # Time
        if result.actual_time_hours is not None:
            icon = "✅" if abs(result.time_error_pct or 0) <= 20 else "⚠️"
            lines.extend([
                "Training Time:",
                f"  Estimated:  {result.estimated_time_hours:.2f} hours",
                f"  Actual:     {result.actual_time_hours:.2f} hours",
                f"  {icon} Error:  {result.time_error_hours:+.2f} hours ({result.time_error_pct:+.1f}%)",
                "",
            ])

        # Cost
        if result.actual_cost is not None and result.estimated_cost is not None:
            icon = "✅" if abs(result.cost_error_pct or 0) <= 20 else "⚠️"
            lines.extend([
                "Cost:",
                f"  Estimated:  ${result.estimated_cost:.2f}",
                f"  Actual:     ${result.actual_cost:.2f}",
                f"  {icon} Error:  {result.cost_error_pct:+.1f}%",
                "",
            ])

        # Grade
        lines.extend([
            f"Overall Accuracy: {result.accuracy_grade}",
            "",
        ])

        # Notes
        if result.notes:
            lines.append("Notes:")
            for note in result.notes:
                lines.append(f"  • {note}")

        return "\n".join(lines)

    @staticmethod
    def from_json(path: str) -> ActualMetrics:
        """Load actual metrics from a JSON file.

        Expected format:
        {
            "peak_memory_gb": 11.2,
            "training_time_hours": 4.5,
            "total_cost": 8.50,
            "gpu_name": "A100-80GB",
            "num_gpus": 1,
            "dataset_size": 50000,
            "epochs": 3,
            "batch_size": 4,
            "seq_length": 2048,
            "method": "qlora",
            "model_name": "meta-llama/Llama-3.1-8B"
        }
        """
        with open(path, "r") as f:
            data = json.load(f)

        return ActualMetrics(
            peak_memory_gb=data.get("peak_memory_gb"),
            training_time_hours=data.get("training_time_hours"),
            total_cost=data.get("total_cost"),
            gpu_name=data.get("gpu_name"),
            num_gpus=data.get("num_gpus", 1),
            dataset_size=data.get("dataset_size"),
            epochs=data.get("epochs"),
            batch_size=data.get("batch_size"),
            seq_length=data.get("seq_length"),
            method=data.get("method"),
            model_name=data.get("model_name"),
            source="json",
            raw_data=data,
        )

    @staticmethod
    def from_wandb(run_path: str) -> ActualMetrics:
        """Load actual metrics from a Weights & Biases run.

        Args:
            run_path: W&B run path (e.g. 'username/project/run_id').

        Requires `wandb` package to be installed.
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb package is required for W&B integration. "
                "Install it with: pip install wandb"
            )

        api = wandb.Api()
        run = api.run(run_path)

        # Extract metrics from W&B run
        summary = run.summary._json_dict
        config = run.config

        # Common metric keys across training frameworks
        peak_memory = (
            summary.get("train/gpu_memory_allocated_gb")
            or summary.get("gpu_memory_usage")
            or summary.get("peak_gpu_memory_gb")
            or summary.get("system/gpu.0.memory_allocated")
        )

        # Convert from MB to GB if needed
        if peak_memory and peak_memory > 200:  # Likely in MB
            peak_memory = peak_memory / 1024

        training_time = (
            summary.get("_wandb", {}).get("runtime")  # seconds
            or summary.get("train/total_time")
            or summary.get("runtime")
        )
        if training_time:
            training_time = training_time / 3600  # Convert to hours

        return ActualMetrics(
            peak_memory_gb=peak_memory,
            training_time_hours=training_time,
            gpu_name=config.get("gpu_type") or config.get("gpu"),
            num_gpus=config.get("num_gpus", 1),
            dataset_size=config.get("dataset_size") or config.get("num_train_samples"),
            epochs=config.get("num_train_epochs") or config.get("epochs"),
            batch_size=config.get("per_device_train_batch_size") or config.get("batch_size"),
            seq_length=config.get("max_seq_length") or config.get("max_length"),
            method=config.get("training_method") or config.get("peft_type"),
            model_name=config.get("model_name_or_path") or config.get("model"),
            source="wandb",
            raw_data={"summary": summary, "config": config},
        )

    @staticmethod
    def from_csv(path: str) -> List[ActualMetrics]:
        """Load multiple actual metrics from a CSV file.

        Expected columns (all optional):
            model_name, method, gpu_name, num_gpus, dataset_size, epochs,
            batch_size, seq_length, peak_memory_gb, training_time_hours, total_cost

        Returns:
            List of ActualMetrics, one per row.
        """
        metrics = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                m = ActualMetrics(
                    peak_memory_gb=float(row["peak_memory_gb"]) if row.get("peak_memory_gb") else None,
                    training_time_hours=float(row["training_time_hours"]) if row.get("training_time_hours") else None,
                    total_cost=float(row["total_cost"]) if row.get("total_cost") else None,
                    gpu_name=row.get("gpu_name"),
                    num_gpus=int(row.get("num_gpus", 1)),
                    dataset_size=int(row["dataset_size"]) if row.get("dataset_size") else None,
                    epochs=int(row["epochs"]) if row.get("epochs") else None,
                    batch_size=int(row["batch_size"]) if row.get("batch_size") else None,
                    seq_length=int(row["seq_length"]) if row.get("seq_length") else None,
                    method=row.get("method"),
                    model_name=row.get("model_name"),
                    source="csv",
                    raw_data=dict(row),
                )
                metrics.append(m)
        return metrics
