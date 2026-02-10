"""ftune — Estimate GPU memory, training time, and costs for LLM fine-tuning.

Quick start:
    >>> from ftune import Estimator
    >>> est = Estimator(model="meta-llama/Llama-3.1-8B", method="qlora", quantization="4bit")
    >>> mem = est.estimate_memory()
    >>> print(f"Total VRAM: {mem.total_gb:.2f} GB")
"""

from ftune.core.models import (
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
    TimeEstimate,
    TrainingConfig,
)
from ftune.estimator import Estimator
from ftune.loader import get_gpu, get_model, list_gpu_names, list_model_names

__version__ = "0.1.0"

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
]
