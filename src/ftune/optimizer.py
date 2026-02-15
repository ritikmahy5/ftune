"""Budget optimizer — find the best training configuration given constraints.

Reverses the logic: instead of "what will this config cost?",
answers "given $X budget and GPU Y, what's the optimal config?"
"""

from __future__ import annotations

from typing import List, Optional

from ftune.core.cost import CostEstimator
from ftune.core.memory import MemoryEstimator
from ftune.core.models import (
    BudgetRecommendation,
    FineTuneMethod,
    LoRATarget,
    OptimizerType,
    Quantization,
    ShardingStrategy,
    TrainingConfig,
)
from ftune.core.time import TimeEstimator
from ftune.loader import get_gpu, get_model, load_gpus


# Search space for configuration optimizer
LORA_RANKS = [8, 16, 32, 64]
BATCH_SIZES = [1, 2, 4, 8]
GRAD_ACCUM_STEPS = [1, 2, 4, 8]


class BudgetOptimizer:
    """Find optimal fine-tuning configuration given budget/hardware constraints.

    Example:
        >>> from ftune.optimizer import BudgetOptimizer
        >>> recs = BudgetOptimizer.optimize(
        ...     model="meta-llama/Llama-3.1-8B",
        ...     budget=20.0,
        ...     gpu="RTX-3090-24GB",
        ...     dataset_size=50000,
        ...     epochs=3,
        ...     priority="cost",
        ... )
        >>> for r in recs:
        ...     print(f"{r.method} rank={r.lora_rank} → ${r.estimated_cost:.2f}")
    """

    @staticmethod
    def optimize(
        model: str,
        dataset_size: int = 50000,
        epochs: int = 3,
        budget: Optional[float] = None,
        gpu: Optional[str] = None,
        num_gpus: int = 1,
        priority: str = "cost",  # "cost", "speed", "quality"
        seq_length: int = 2048,
    ) -> List[BudgetRecommendation]:
        """Find optimal configurations within constraints.

        Args:
            model: Model name.
            dataset_size: Training dataset size.
            epochs: Number of epochs.
            budget: Max budget in USD (None = no budget constraint).
            gpu: Specific GPU to target (None = search all).
            num_gpus: Number of GPUs available.
            priority: Optimization priority.
            seq_length: Sequence length.

        Returns:
            List of BudgetRecommendation sorted by priority.
        """
        model_spec = get_model(model)
        cost_estimator = CostEstimator()

        # Determine which GPUs to consider
        if gpu:
            gpu_specs = {gpu: get_gpu(gpu)}
        else:
            gpu_specs = load_gpus()

        candidates = []

        # Configuration search space
        methods = [
            (FineTuneMethod.QLORA, Quantization.INT4),
            (FineTuneMethod.QLORA, Quantization.INT8),
            (FineTuneMethod.LORA, Quantization.NONE),
        ]

        # Only try full fine-tune if we have enough GPUs
        if num_gpus >= 4 or model_spec.parameters < 3e9:
            methods.append((FineTuneMethod.FULL, Quantization.NONE))

        targets = [LoRATarget.ATTENTION, LoRATarget.ALL_LINEAR]
        optimizers = [OptimizerType.ADAMW, OptimizerType.ADAM_8BIT]

        for method, quant in methods:
            for rank in LORA_RANKS if method != FineTuneMethod.FULL else [0]:
                for target in targets if method != FineTuneMethod.FULL else [LoRATarget.ATTENTION]:
                    for bs in BATCH_SIZES:
                        for ga in GRAD_ACCUM_STEPS:
                            for opt in optimizers:
                                for flash in [True, False]:
                                    for gc in [True, False]:
                                        # Build config
                                        config = TrainingConfig(
                                            model=model,
                                            method=method,
                                            quantization=quant,
                                            batch_size=bs,
                                            seq_length=seq_length,
                                            gradient_checkpointing=gc,
                                            optimizer=opt,
                                            lora_rank=max(rank, 1),
                                            lora_alpha=rank * 2 if rank > 0 else 32,
                                            lora_target=target,
                                            flash_attention=flash,
                                            sharding=ShardingStrategy.NONE,
                                            num_gpus=num_gpus,
                                        )

                                        # Estimate memory
                                        mem_est = MemoryEstimator(model_spec, config)
                                        try:
                                            mem = mem_est.estimate()
                                        except Exception:
                                            continue

                                        # Check against each GPU
                                        for gpu_name, gpu_spec in gpu_specs.items():
                                            if mem.total_gb > gpu_spec.vram_gb:
                                                continue

                                            # Estimate time
                                            time_est = TimeEstimator(model_spec, config)
                                            try:
                                                time_result = time_est.estimate(
                                                    gpu=gpu_spec,
                                                    dataset_size=dataset_size,
                                                    epochs=epochs,
                                                    num_gpus=num_gpus,
                                                )
                                            except Exception:
                                                continue

                                            # Estimate cost
                                            cost_estimates = cost_estimator.estimate_for_gpu(
                                                gpu_name, time_result.total_hours
                                            )
                                            if not cost_estimates:
                                                # Use a default rate if no pricing
                                                est_cost = time_result.total_hours * 2.0
                                                provider = "Unknown"
                                            else:
                                                cheapest = cost_estimates[0]
                                                est_cost = cheapest.total_cost
                                                provider = cheapest.provider

                                            # Budget filter
                                            if budget and est_cost > budget:
                                                continue

                                            # Quality score (higher rank + more target modules = better)
                                            quality_score = (rank or 1) * (
                                                1 if target == LoRATarget.ATTENTION else 2
                                            )

                                            notes = []
                                            if flash:
                                                notes.append("FlashAttention-2 enabled")
                                            if gc:
                                                notes.append("Gradient checkpointing ON")
                                            if opt == OptimizerType.ADAM_8BIT:
                                                notes.append("8-bit Adam (75% less optimizer memory)")

                                            candidates.append((
                                                BudgetRecommendation(
                                                    method=method.value,
                                                    quantization=quant.value,
                                                    lora_rank=rank,
                                                    lora_target=target.value,
                                                    batch_size=bs,
                                                    gradient_accumulation=ga,
                                                    gradient_checkpointing=gc,
                                                    optimizer=opt.value,
                                                    flash_attention=flash,
                                                    sharding=ShardingStrategy.NONE.value,
                                                    estimated_memory_gb=mem.total_gb,
                                                    estimated_hours=time_result.total_hours,
                                                    estimated_cost=round(est_cost, 2),
                                                    gpu=gpu_name,
                                                    provider=provider,
                                                    num_gpus=num_gpus,
                                                    fits=True,
                                                    priority=priority,
                                                    notes=notes,
                                                ),
                                                est_cost,
                                                time_result.total_hours,
                                                quality_score,
                                            ))

        # Sort by priority
        if priority == "cost":
            candidates.sort(key=lambda x: (x[1], x[2]))
        elif priority == "speed":
            candidates.sort(key=lambda x: (x[2], x[1]))
        elif priority == "quality":
            candidates.sort(key=lambda x: (-x[3], x[1]))

        # Deduplicate — keep top diverse configs
        seen = set()
        results = []
        for rec, cost, hours, quality in candidates:
            key = (rec.method, rec.quantization, rec.lora_rank, rec.gpu, rec.flash_attention)
            if key not in seen:
                seen.add(key)
                results.append(rec)
            if len(results) >= 10:
                break

        return results

    @staticmethod
    def format_recommendations(recs: List[BudgetRecommendation]) -> str:
        """Format recommendations as readable text."""
        if not recs:
            return "No configurations found within the given constraints."

        lines = [
            "╭────────────────────────────────────────────────╮",
            "│    🎯 ftune Budget Optimizer — Recommendations  │",
            "╰────────────────────────────────────────────────╯",
            "",
        ]

        for i, r in enumerate(recs[:5], 1):
            rank_str = f", rank={r.lora_rank}" if r.lora_rank > 0 else ""
            lines.extend([
                f"  #{i}: {r.method.upper()} {r.quantization}{rank_str}",
                f"      GPU: {r.gpu} ({r.provider}) | {r.num_gpus} GPU(s)",
                f"      Batch: {r.batch_size} × {r.gradient_accumulation} accum | "
                f"Optimizer: {r.optimizer}",
                f"      Memory: {r.estimated_memory_gb:.1f} GB | "
                f"Time: {r.estimated_hours:.1f}h | "
                f"Cost: ${r.estimated_cost:.2f}",
            ])
            if r.notes:
                lines.append(f"      💡 {', '.join(r.notes)}")
            lines.append("")

        return "\n".join(lines)
