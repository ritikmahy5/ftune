"""Memory estimation engine for LLM fine-tuning.

Calculates GPU VRAM requirements for full fine-tuning, LoRA, and QLoRA
with support for FlashAttention-2 and ZeRO/FSDP sharding strategies.
"""

from __future__ import annotations

from typing import List

from ftune.core.models import (
    FineTuneMethod,
    GPUFit,
    GPUSpec,
    LoRATarget,
    MemoryBreakdown,
    ModelSpec,
    OptimizerType,
    Quantization,
    ShardingStrategy,
    TrainingConfig,
)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

BYTES_PER_PARAM = {
    "float32": 4, "fp32": 4,
    "float16": 2, "fp16": 2,
    "bfloat16": 2, "bf16": 2,
}

OPTIMIZER_BYTES_PER_PARAM = {
    OptimizerType.ADAM: 8,       # 2 states × 4 bytes (fp32)
    OptimizerType.ADAMW: 8,
    OptimizerType.SGD: 4,       # 1 state × 4 bytes
    OptimizerType.ADAM_8BIT: 2,  # 2 states × 1 byte
    OptimizerType.ADAFACTOR: 4,  # row + col factors ≈ 4 bytes
}

QUANTIZATION_BYTES = {
    Quantization.NONE: None,
    Quantization.INT8: 1.0,
    Quantization.INT4: 0.5,
}

QUANTIZATION_OVERHEAD_FRACTION = {
    Quantization.INT8: 0.01,
    Quantization.INT4: 0.02,
}

# Activation factors
ACTIVATION_FACTOR_WITH_CHECKPOINTING = 2.0
ACTIVATION_FACTOR_WITHOUT_CHECKPOINTING = 10.0

# FlashAttention-2 reduces activation memory for attention computation
# by ~40-60% compared to standard attention (no materialized attention matrix).
# We model this as a multiplier on activation memory.
FLASH_ATTENTION_ACTIVATION_REDUCTION = 0.5  # 50% reduction in activation memory

# CUDA overhead
CUDA_OVERHEAD_FRACTION = 0.15

BYTES_TO_GB = 1 / (1024**3)


# ─────────────────────────────────────────────────────────
# ZeRO/FSDP Sharding Factors
# ─────────────────────────────────────────────────────────
# Each strategy shards certain components across num_gpus.
# The factor represents what fraction of the component
# each GPU holds: 1.0 = full copy, 1/N = sharded.

def _get_sharding_factors(strategy: ShardingStrategy, num_gpus: int) -> dict:
    """Get per-component sharding factors.

    Returns dict with keys: 'weights', 'gradients', 'optimizer'
    Each value is the fraction of that component held per GPU.
    """
    n = max(num_gpus, 1)

    if strategy == ShardingStrategy.NONE:
        return {"weights": 1.0, "gradients": 1.0, "optimizer": 1.0}

    elif strategy == ShardingStrategy.ZERO_1:
        # ZeRO-1: shard only optimizer states
        return {"weights": 1.0, "gradients": 1.0, "optimizer": 1.0 / n}

    elif strategy in (ShardingStrategy.ZERO_2, ShardingStrategy.FSDP_SHARD_GRAD):
        # ZeRO-2 / FSDP shard-grad: shard optimizer + gradients
        return {"weights": 1.0, "gradients": 1.0 / n, "optimizer": 1.0 / n}

    elif strategy in (ShardingStrategy.ZERO_3, ShardingStrategy.FSDP):
        # ZeRO-3 / FSDP full: shard everything
        return {"weights": 1.0 / n, "gradients": 1.0 / n, "optimizer": 1.0 / n}

    return {"weights": 1.0, "gradients": 1.0, "optimizer": 1.0}


class MemoryEstimator:
    """Estimates GPU VRAM required for LLM fine-tuning.

    Supports:
    - Full fine-tuning, LoRA, QLoRA
    - FlashAttention-2 activation reduction
    - ZeRO Stages 1/2/3 and FSDP sharding

    Args:
        model_spec: Model architecture specifications.
        config: Training configuration.
    """

    def __init__(self, model_spec: ModelSpec, config: TrainingConfig) -> None:
        self.model = model_spec
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate internal consistency."""
        if self.config.method == FineTuneMethod.QLORA and self.config.quantization == Quantization.NONE:
            self.config.quantization = Quantization.INT4

        if self.config.method == FineTuneMethod.FULL and self.config.quantization != Quantization.NONE:
            raise ValueError(
                "Full fine-tuning does not support quantization. "
                "Use method='lora' or method='qlora' with quantization."
            )

        if self.config.seq_length > self.model.max_seq_length:
            raise ValueError(
                f"Requested seq_length ({self.config.seq_length}) exceeds model's "
                f"max_seq_length ({self.model.max_seq_length})."
            )

    # ─────────────────────────────────────────────────────
    # Bytes per param helpers
    # ─────────────────────────────────────────────────────

    @property
    def _dtype_bytes(self) -> int:
        return BYTES_PER_PARAM.get(self.model.default_dtype, 2)

    @property
    def _base_model_bytes_per_param(self) -> float:
        quant = self.config.quantization
        if quant != Quantization.NONE:
            return QUANTIZATION_BYTES[quant]
        return float(self._dtype_bytes)

    # ─────────────────────────────────────────────────────
    # LoRA
    # ─────────────────────────────────────────────────────

    def _count_lora_target_modules(self) -> int:
        target = self.config.lora_target
        if target == LoRATarget.ATTENTION:
            return 2
        elif target == LoRATarget.ATTENTION_ALL:
            return 4
        elif target == LoRATarget.ALL_LINEAR:
            return 7
        return 2

    def _compute_lora_params(self) -> int:
        modules_per_layer = self._count_lora_target_modules()
        rank = self.config.lora_rank
        hidden = self.model.hidden_size
        params_per_module = 2 * hidden * rank
        return params_per_module * modules_per_layer * self.model.num_layers

    # ─────────────────────────────────────────────────────
    # Memory components
    # ─────────────────────────────────────────────────────

    def _model_weights_bytes(self) -> float:
        """Base model weight memory."""
        base = self.model.parameters * self._base_model_bytes_per_param
        quant = self.config.quantization
        if quant in QUANTIZATION_OVERHEAD_FRACTION:
            base += self.model.parameters * self._base_model_bytes_per_param * QUANTIZATION_OVERHEAD_FRACTION[quant]
        return base

    def _trainable_params_bytes(self) -> float:
        """Trainable parameter memory (LoRA adapters etc)."""
        if self.config.method == FineTuneMethod.FULL:
            return 0.0
        return self._compute_lora_params() * 2  # bf16

    def _gradient_bytes(self) -> float:
        """Gradient memory for trainable params."""
        return self._get_trainable_param_count() * 2  # bf16

    def _optimizer_state_bytes(self) -> float:
        """Optimizer state memory."""
        trainable = self._get_trainable_param_count()
        return trainable * OPTIMIZER_BYTES_PER_PARAM[self.config.optimizer]

    def _activation_bytes(self) -> float:
        """Activation memory during forward/backward.

        FlashAttention-2 avoids materializing the full N×N attention matrix,
        reducing activation memory by ~50%. This is the single biggest
        memory optimization available for long sequences.
        """
        batch = self.config.batch_size
        seq = self.config.seq_length
        hidden = self.model.hidden_size
        layers = self.model.num_layers
        dtype_bytes = 2  # fp16/bf16

        factor = (
            ACTIVATION_FACTOR_WITH_CHECKPOINTING
            if self.config.gradient_checkpointing
            else ACTIVATION_FACTOR_WITHOUT_CHECKPOINTING
        )

        base_activation = batch * seq * hidden * layers * dtype_bytes * factor

        # FlashAttention-2 reduces activation memory
        if self.config.flash_attention:
            base_activation *= FLASH_ATTENTION_ACTIVATION_REDUCTION

        return base_activation

    def _get_trainable_param_count(self) -> int:
        if self.config.method == FineTuneMethod.FULL:
            return self.model.parameters
        return self._compute_lora_params()

    # ─────────────────────────────────────────────────────
    # Main estimation
    # ─────────────────────────────────────────────────────

    def estimate(self) -> MemoryBreakdown:
        """Calculate complete per-GPU VRAM breakdown.

        When sharding is enabled, model weights, gradients, and optimizer
        states are divided across GPUs according to the ZeRO/FSDP strategy.
        Activations are NOT sharded (each GPU computes its own).
        """
        # Raw (unsharded) component sizes
        model_weights = self._model_weights_bytes()
        trainable_mem = self._trainable_params_bytes()
        gradients = self._gradient_bytes()
        optimizer = self._optimizer_state_bytes()
        activations = self._activation_bytes()

        # Apply sharding factors (per-GPU memory)
        sharding = _get_sharding_factors(self.config.sharding, self.config.num_gpus)

        model_weights_per_gpu = model_weights * sharding["weights"]
        trainable_per_gpu = trainable_mem * sharding["weights"]
        gradients_per_gpu = gradients * sharding["gradients"]
        optimizer_per_gpu = optimizer * sharding["optimizer"]
        # Activations are NOT sharded — each GPU has its own copy
        activations_per_gpu = activations

        # ZeRO-3/FSDP need communication buffers for all-gather
        comm_buffer = 0.0
        if self.config.sharding in (
            ShardingStrategy.ZERO_3, ShardingStrategy.FSDP
        ) and self.config.num_gpus > 1:
            # ~5% of model size for all-gather buffers
            comm_buffer = model_weights * 0.05

        subtotal = (
            model_weights_per_gpu + trainable_per_gpu + gradients_per_gpu
            + optimizer_per_gpu + activations_per_gpu + comm_buffer
        )
        overhead = subtotal * CUDA_OVERHEAD_FRACTION
        total = subtotal + overhead

        trainable_params = self._get_trainable_param_count()
        trainable_pct = (trainable_params / self.model.parameters) * 100

        return MemoryBreakdown(
            model_weights_gb=round(model_weights_per_gpu * BYTES_TO_GB, 3),
            trainable_params_gb=round(trainable_per_gpu * BYTES_TO_GB, 3),
            gradients_gb=round(gradients_per_gpu * BYTES_TO_GB, 3),
            optimizer_states_gb=round(optimizer_per_gpu * BYTES_TO_GB, 3),
            activations_gb=round(activations_per_gpu * BYTES_TO_GB, 3),
            overhead_gb=round((overhead + comm_buffer) * BYTES_TO_GB, 3),
            total_gb=round(total * BYTES_TO_GB, 3),
            total_params=self.model.parameters,
            trainable_params=trainable_params,
            trainable_percentage=round(trainable_pct, 4),
            method=self.config.method.value,
            quantization=self.config.quantization.value,
            sharding=self.config.sharding.value,
        )

    def check_gpu_fit(self, gpus: List[GPUSpec]) -> List[GPUFit]:
        """Check which GPUs can fit the estimated memory."""
        mem = self.estimate()
        results = []
        for gpu in gpus:
            utilization = (mem.total_gb / gpu.vram_gb) * 100
            headroom = gpu.vram_gb - mem.total_gb
            results.append(GPUFit(
                gpu_name=gpu.name,
                vram_gb=gpu.vram_gb,
                required_gb=mem.total_gb,
                fits=mem.total_gb <= gpu.vram_gb,
                utilization_percent=round(utilization, 1),
                headroom_gb=round(headroom, 2),
            ))
        results.sort(key=lambda g: (not g.fits, -g.headroom_gb))
        return results
