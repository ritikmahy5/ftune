"""Training time estimation engine for LLM fine-tuning.

Estimates wall-clock training time based on model size, dataset,
GPU compute capacity, and training configuration.
"""

from __future__ import annotations

from ftune.core.models import (
    FineTuneMethod,
    GPUSpec,
    ModelSpec,
    TimeEstimate,
    TrainingConfig,
)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

# FLOPs per token for a forward + backward pass
# Approximation: ~6 × num_params per token (2x forward, 4x backward)
FLOPS_PER_TOKEN_FACTOR = 6

# Model FLOPs Utilization — fraction of theoretical GPU TFLOPS achieved
# Conservative defaults based on empirical fine-tuning benchmarks
DEFAULT_MFU = {
    "full": 0.35,    # Full fine-tuning baseline MFU
    "lora": 0.35,    # LoRA achieves similar MFU to full
    "qlora": 0.30,   # QLoRA has quantization/dequantization overhead
}

# Mixed precision speedup factor (bf16/fp16 vs fp32)
MIXED_PRECISION_SPEEDUP = 1.0  # Already accounted for in TFLOPS specs


class TimeEstimator:
    """Estimates training wall-clock time for LLM fine-tuning.

    Uses FLOPs-based estimation with Model FLOPs Utilization (MFU)
    to predict real-world training duration.

    Args:
        model_spec: Model architecture specifications.
        config: Training configuration.
    """

    def __init__(self, model_spec: ModelSpec, config: TrainingConfig) -> None:
        self.model = model_spec
        self.config = config

    def _compute_total_tokens(self, dataset_size: int, epochs: int, avg_seq_length: int) -> int:
        """Calculate total tokens processed during training.

        Args:
            dataset_size: Number of samples in the dataset.
            epochs: Number of training epochs.
            avg_seq_length: Average sequence length per sample.

        Returns:
            Total token count across all epochs.
        """
        return dataset_size * avg_seq_length * epochs

    def _compute_steps(
        self, dataset_size: int, epochs: int, gradient_accumulation_steps: int = 1
    ) -> tuple:
        """Calculate training steps.

        Args:
            dataset_size: Number of samples.
            epochs: Number of epochs.
            gradient_accumulation_steps: Gradient accumulation steps.

        Returns:
            Tuple of (steps_per_epoch, total_steps).
        """
        effective_batch = self.config.batch_size * gradient_accumulation_steps
        steps_per_epoch = max(1, dataset_size // effective_batch)
        total_steps = steps_per_epoch * epochs
        return steps_per_epoch, total_steps

    def _compute_flops_per_token(self) -> float:
        """Calculate FLOPs needed per token (forward + backward).

        For full fine-tuning: 6 × num_params
        For LoRA: forward uses full model, backward only updates adapters
            Effective FLOPs ≈ 2 × num_params (forward) + 4 × trainable_params (backward)
        """
        total_params = self.model.parameters
        method = self.config.method

        if method == FineTuneMethod.FULL:
            return FLOPS_PER_TOKEN_FACTOR * total_params

        # LoRA/QLoRA: forward pass is full, backward is partial
        # But in practice, backward still touches most of the computation graph
        # Empirically, LoRA is ~70-80% the FLOPs of full fine-tuning
        lora_efficiency = 0.75
        return FLOPS_PER_TOKEN_FACTOR * total_params * lora_efficiency

    def _get_gpu_tflops(self, gpu: GPUSpec) -> float:
        """Get effective TFLOPS for the GPU based on dtype.

        Uses bf16 TFLOPS as default since most fine-tuning uses mixed precision.

        Returns:
            Effective TFLOPS (in teraFLOPS).
        """
        dtype = self.model.default_dtype
        if dtype in ("bfloat16", "bf16"):
            return gpu.bf16_tflops
        elif dtype in ("float16", "fp16"):
            return gpu.fp16_tflops
        return gpu.fp32_tflops

    def _get_mfu(self) -> float:
        """Get Model FLOPs Utilization for the training method."""
        return DEFAULT_MFU.get(self.config.method.value, 0.35)

    def estimate(
        self,
        gpu: GPUSpec,
        dataset_size: int,
        epochs: int = 3,
        num_gpus: int = 1,
        avg_seq_length: int | None = None,
        gradient_accumulation_steps: int = 1,
        mfu_override: float | None = None,
    ) -> TimeEstimate:
        """Estimate training time.

        Args:
            gpu: GPU specification to estimate for.
            dataset_size: Number of training samples.
            epochs: Number of training epochs.
            num_gpus: Number of GPUs (data parallel).
            avg_seq_length: Average tokens per sample. Defaults to config.seq_length.
            gradient_accumulation_steps: Gradient accumulation steps.
            mfu_override: Override default MFU if you have empirical data.

        Returns:
            TimeEstimate with detailed timing breakdown.
        """
        if avg_seq_length is None:
            avg_seq_length = self.config.seq_length

        # Total tokens
        total_tokens = self._compute_total_tokens(dataset_size, epochs, avg_seq_length)

        # Steps
        steps_per_epoch, total_steps = self._compute_steps(
            dataset_size, epochs, gradient_accumulation_steps
        )

        # FLOPs calculation
        flops_per_token = self._compute_flops_per_token()
        total_flops = flops_per_token * total_tokens

        # GPU compute capacity
        gpu_tflops = self._get_gpu_tflops(gpu)
        mfu = mfu_override if mfu_override is not None else self._get_mfu()

        # Effective compute: TFLOPS × MFU × num_gpus
        effective_tflops = gpu_tflops * mfu * num_gpus
        effective_flops_per_sec = effective_tflops * 1e12  # Convert to FLOPS

        # Multi-GPU scaling efficiency (accounts for communication overhead)
        if num_gpus > 1:
            # ~90% scaling efficiency per additional GPU (empirical)
            scaling_efficiency = 0.9 ** (num_gpus - 1)
            # But floor at 0.7 (diminishing returns plateau)
            scaling_efficiency = max(scaling_efficiency, 0.7)
            effective_flops_per_sec *= scaling_efficiency

        # Time calculation
        total_seconds = total_flops / effective_flops_per_sec if effective_flops_per_sec > 0 else 0
        total_hours = total_seconds / 3600
        hours_per_epoch = total_hours / epochs if epochs > 0 else 0

        return TimeEstimate(
            total_hours=round(total_hours, 2),
            hours_per_epoch=round(hours_per_epoch, 2),
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
            total_tokens=total_tokens,
            total_tflops=round(total_flops / 1e12, 2),
            gpu_name=gpu.name,
            num_gpus=num_gpus,
            mfu=mfu,
            dataset_size=dataset_size,
            epochs=epochs,
        )
