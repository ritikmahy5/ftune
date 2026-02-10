"""High-level Estimator API for ftune.

This module provides the main user-facing class that wraps the core
estimation engines with a simple, ergonomic interface.
"""

from __future__ import annotations

from typing import List, Optional, Union

from ftune.core.cost import CostEstimator
from ftune.core.memory import MemoryEstimator
from ftune.core.models import (
    CostComparison,
    CostEstimate,
    FineTuneMethod,
    GPUFit,
    LoRATarget,
    MemoryBreakdown,
    OptimizerType,
    Quantization,
    TimeEstimate,
    TrainingConfig,
)
from ftune.core.time import TimeEstimator
from ftune.loader import get_gpu, get_model, load_gpus


class Estimator:
    """Estimate GPU memory for LLM fine-tuning.

    Provides a simple interface to estimate VRAM requirements and check
    GPU compatibility for various fine-tuning configurations.

    Args:
        model: Model name from the database (e.g. 'meta-llama/Llama-3.1-8B').
        method: Fine-tuning method ('full', 'lora', 'qlora').
        quantization: Quantization level ('none', '4bit', '8bit').
        batch_size: Per-device training batch size.
        seq_length: Maximum sequence length for training.
        gradient_checkpointing: Whether to use gradient/activation checkpointing.
        optimizer: Optimizer type ('adamw', 'adam', 'sgd', 'adam_8bit').
        lora_rank: LoRA rank (r parameter).
        lora_alpha: LoRA alpha scaling parameter.
        lora_target: LoRA target modules ('attention', 'attention_all', 'all_linear').

    Example:
        >>> est = Estimator(
        ...     model="meta-llama/Llama-3.1-8B",
        ...     method="qlora",
        ...     quantization="4bit",
        ...     lora_rank=16,
        ...     batch_size=4,
        ... )
        >>> mem = est.estimate_memory()
        >>> print(f"Total VRAM: {mem.total_gb:.2f} GB")
    """

    def __init__(
        self,
        model: str,
        method: Union[str, FineTuneMethod] = "lora",
        quantization: Union[str, Quantization] = "none",
        batch_size: int = 4,
        seq_length: int = 2048,
        gradient_checkpointing: bool = True,
        optimizer: Union[str, OptimizerType] = "adamw",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target: Union[str, LoRATarget] = "attention",
    ) -> None:
        # Resolve model spec from database
        self.model_spec = get_model(model)

        # Build training config
        self.config = TrainingConfig(
            model=model,
            method=FineTuneMethod(method) if isinstance(method, str) else method,
            quantization=Quantization(quantization) if isinstance(quantization, str) else quantization,
            batch_size=batch_size,
            seq_length=seq_length,
            gradient_checkpointing=gradient_checkpointing,
            optimizer=OptimizerType(optimizer) if isinstance(optimizer, str) else optimizer,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target=LoRATarget(lora_target) if isinstance(lora_target, str) else lora_target,
        )

        # Initialize core estimators
        self._memory_estimator = MemoryEstimator(self.model_spec, self.config)
        self._time_estimator = TimeEstimator(self.model_spec, self.config)
        self._cost_estimator = CostEstimator()

    def estimate_memory(self) -> MemoryBreakdown:
        """Estimate total GPU VRAM required for training.

        Returns:
            MemoryBreakdown with component-level VRAM usage in GB.
        """
        return self._memory_estimator.estimate()

    def check_gpu_fit(self, gpu_names: Optional[List[str]] = None) -> List[GPUFit]:
        """Check which GPUs can handle this training configuration.

        Args:
            gpu_names: Optional list of specific GPU names to check.
                If None, checks against all GPUs in the database.

        Returns:
            List of GPUFit results, sorted with best fits first.
        """
        all_gpus = load_gpus()

        if gpu_names:
            from ftune.loader import get_gpu
            gpus = [get_gpu(name) for name in gpu_names]
        else:
            gpus = list(all_gpus.values())

        return self._memory_estimator.check_gpu_fit(gpus)

    def estimate_time(
        self,
        gpu: str = "A100-80GB",
        dataset_size: int = 50000,
        epochs: int = 3,
        num_gpus: int = 1,
        avg_seq_length: int | None = None,
        gradient_accumulation_steps: int = 1,
        mfu_override: float | None = None,
    ) -> TimeEstimate:
        """Estimate training wall-clock time.

        Args:
            gpu: GPU name from database (e.g. 'A100-80GB', 'RTX-4090-24GB').
            dataset_size: Number of training samples.
            epochs: Number of training epochs.
            num_gpus: Number of GPUs (data parallel scaling).
            avg_seq_length: Average tokens per sample. Defaults to config.seq_length.
            gradient_accumulation_steps: Gradient accumulation steps.
            mfu_override: Override default Model FLOPs Utilization.

        Returns:
            TimeEstimate with timing breakdown.
        """
        gpu_spec = get_gpu(gpu)
        return self._time_estimator.estimate(
            gpu=gpu_spec,
            dataset_size=dataset_size,
            epochs=epochs,
            num_gpus=num_gpus,
            avg_seq_length=avg_seq_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mfu_override=mfu_override,
        )

    def estimate_time_all_gpus(
        self,
        dataset_size: int = 50000,
        epochs: int = 3,
        num_gpus: int = 1,
        avg_seq_length: int | None = None,
    ) -> List[TimeEstimate]:
        """Estimate training time across all compatible GPUs.

        Only includes GPUs that can fit the training memory requirement.

        Returns:
            List of TimeEstimate sorted by total_hours (fastest first).
        """
        fits = self.check_gpu_fit()
        compatible = [f for f in fits if f.fits]
        results = []

        for gpu_fit in compatible:
            time_est = self.estimate_time(
                gpu=gpu_fit.gpu_name,
                dataset_size=dataset_size,
                epochs=epochs,
                num_gpus=num_gpus,
                avg_seq_length=avg_seq_length,
            )
            results.append(time_est)

        results.sort(key=lambda t: t.total_hours)
        return results

    def estimate_costs(
        self,
        gpu: str = "A100-80GB",
        training_hours: float | None = None,
        dataset_size: int = 50000,
        epochs: int = 3,
        num_gpus: int = 1,
        include_spot: bool = True,
    ) -> CostComparison:
        """Estimate cloud costs across all providers.

        If training_hours is not provided, it will be calculated automatically
        using estimate_time().

        Args:
            gpu: GPU name to compare pricing for.
            training_hours: Pre-computed training hours. If None, calculated.
            dataset_size: Number of training samples (used if training_hours is None).
            epochs: Training epochs (used if training_hours is None).
            num_gpus: Number of GPUs.
            include_spot: Include spot/preemptible pricing.

        Returns:
            CostComparison with all provider estimates.
        """
        if training_hours is None:
            time_est = self.estimate_time(
                gpu=gpu, dataset_size=dataset_size, epochs=epochs, num_gpus=num_gpus,
            )
            training_hours = time_est.total_hours

        return self._cost_estimator.quick_estimate(
            gpu_name=gpu, training_hours=training_hours
        )

    def full_comparison(
        self,
        dataset_size: int = 50000,
        epochs: int = 3,
        num_gpus: int = 1,
    ) -> CostComparison:
        """Full cost comparison across all compatible GPUs and providers.

        Estimates training time for each compatible GPU, then compares
        costs across all providers. This gives the most comprehensive view.

        Returns:
            CostComparison with the cheapest options highlighted.
        """
        time_estimates = self.estimate_time_all_gpus(
            dataset_size=dataset_size, epochs=epochs, num_gpus=num_gpus,
        )

        compatible_gpus = [t.gpu_name for t in time_estimates]
        hours_per_gpu = {t.gpu_name: t.total_hours for t in time_estimates}

        return self._cost_estimator.compare_all(
            compatible_gpus=compatible_gpus,
            training_hours_per_gpu=hours_per_gpu,
            num_gpus=num_gpus,
        )

    def summary(self) -> str:
        """Generate a text summary of the memory estimate.

        Returns:
            Human-readable summary string.
        """
        from ftune.utils.formatting import format_params, format_percentage

        mem = self.estimate_memory()
        fits = self.check_gpu_fit()

        compatible = [g for g in fits if g.fits]
        incompatible = [g for g in fits if not g.fits]

        lines = [
            f"Model: {self.model_spec.name} ({format_params(self.model_spec.parameters)} params)",
            f"Method: {mem.method.upper()}"
            + (f" (rank={self.config.lora_rank})" if mem.method != "full" else "")
            + (f", {mem.quantization}" if mem.quantization != "none" else ""),
            f"Batch size: {self.config.batch_size}, Seq length: {self.config.seq_length}",
            f"Gradient checkpointing: {'ON' if self.config.gradient_checkpointing else 'OFF'}",
            "",
            "Memory Breakdown:",
            f"  Model weights:    {mem.model_weights_gb:>8.2f} GB",
            f"  LoRA adapters:    {mem.trainable_params_gb:>8.2f} GB",
            f"  Gradients:        {mem.gradients_gb:>8.2f} GB",
            f"  Optimizer states: {mem.optimizer_states_gb:>8.2f} GB",
            f"  Activations:      {mem.activations_gb:>8.2f} GB",
            f"  CUDA overhead:    {mem.overhead_gb:>8.2f} GB",
            f"  {'─' * 36}",
            f"  TOTAL:            {mem.total_gb:>8.2f} GB",
            "",
            f"Trainable params: {format_params(mem.trainable_params)} "
            f"({format_percentage(mem.trainable_percentage)})",
            "",
        ]

        if compatible:
            lines.append("✅ Fits on: " + ", ".join(
                f"{g.gpu_name} ({g.utilization_percent:.0f}%)" for g in compatible
            ))
        if incompatible:
            lines.append("❌ Too large for: " + ", ".join(g.gpu_name for g in incompatible))

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Estimator(model='{self.config.model}', method='{self.config.method.value}', "
            f"quantization='{self.config.quantization.value}')"
        )
