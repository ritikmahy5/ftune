"""ftune — Estimate GPU memory, training time, and costs for LLM fine-tuning.

Quick start:
    >>> from ftuneai import Estimator
    >>> est = Estimator(model="meta-llama/Llama-3.1-8B", method="qlora", quantization="4bit")
    >>> mem = est.estimate_memory()
    >>> print(f"Total VRAM: {mem.total_gb:.2f} GB")
"""

from ftuneai.calibration import Calibrator
from ftuneai.core.models import (
    BudgetRecommendation,
    CalibrationResult,
    CostComparison,
    CostEstimate,
    FineTuneMethod,
    GPUFit,
    GPUSpec,
    LoRATarget,
    MemoryBreakdown,
    ModelSpec,
    OptimizerType,
    Quantization,
    ShardingStrategy,
    TimeEstimate,
    TrainingConfig,
)
from ftuneai.estimator import Estimator
from ftuneai.hub import resolve_model_from_hub
from ftuneai.loader import get_gpu, get_model, list_gpu_names, list_model_names
from ftuneai.optimizer import BudgetOptimizer
from ftuneai.validation import ActualMetrics, ValidationResult, Validator

__version__ = "0.2.0"

__all__ = [
    # Main API
    "Estimator",
    # Data models
    "TrainingConfig",
    "MemoryBreakdown",
    "TimeEstimate",
    "CostEstimate",
    "CostComparison",
    "GPUFit",
    "ModelSpec",
    "GPUSpec",
    # Enums
    "FineTuneMethod",
    "Quantization",
    "OptimizerType",
    "LoRATarget",
    # Data loaders
    "get_model",
    "get_gpu",
    "list_model_names",
    "list_gpu_names",
    # Enums
    "ShardingStrategy",
    # HuggingFace Hub
    "resolve_model_from_hub",
    # Validation
    "Validator",
    "ActualMetrics",
    "ValidationResult",
    # Calibration
    "Calibrator",
    "CalibrationResult",
    # Optimizer
    "BudgetOptimizer",
    "BudgetRecommendation",
]
