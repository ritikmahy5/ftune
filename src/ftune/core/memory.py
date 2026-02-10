"""Memory estimation engine for LLM fine-tuning.

Calculates GPU VRAM requirements for full fine-tuning, LoRA, and QLoRA
based on model architecture, training configuration, and optimization settings.
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
    TrainingConfig,
)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

BYTES_PER_PARAM = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
}

# Optimizer memory: bytes per parameter for each state
OPTIMIZER_BYTES_PER_PARAM = {
    OptimizerType.ADAM: 8,       # 2 states (m, v) × 4 bytes (fp32)
    OptimizerType.ADAMW: 8,     # Same as Adam
    OptimizerType.SGD: 4,       # 1 state (momentum) × 4 bytes
    OptimizerType.ADAM_8BIT: 2,  # 2 states × 1 byte (int8)
    OptimizerType.ADAFACTOR: 4,  # Row + col factors ≈ 4 bytes effective
}

# Quantization: effective bytes per parameter
QUANTIZATION_BYTES = {
    Quantization.NONE: None,  # Use model dtype
    Quantization.INT8: 1.0,
    Quantization.INT4: 0.5,
}

# Quantization overhead: block-wise scaling constants as fraction of base
QUANTIZATION_OVERHEAD_FRACTION = {
    Quantization.INT8: 0.01,
    Quantization.INT4: 0.02,
}

# Activation memory factor (multiplied by per-layer activation size)
# These are empirical estimates from real training runs
ACTIVATION_FACTOR_WITH_CHECKPOINTING = 2.0
ACTIVATION_FACTOR_WITHOUT_CHECKPOINTING = 10.0

# CUDA overhead as fraction of total (kernels, fragmentation, buffers)
CUDA_OVERHEAD_FRACTION = 0.15

BYTES_TO_GB = 1 / (1024**3)


class MemoryEstimator:
    """Estimates GPU VRAM required for LLM fine-tuning.

    Supports full fine-tuning, LoRA, and QLoRA methods with detailed
    component-level memory breakdowns.

    Args:
        model_spec: Model architecture specifications.
        config: Training configuration (method, batch size, etc).
    """

    def __init__(self, model_spec: ModelSpec, config: TrainingConfig) -> None:
        self.model = model_spec
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate that the configuration is internally consistent."""
        if self.config.method == FineTuneMethod.QLORA and self.config.quantization == Quantization.NONE:
            # QLoRA requires quantization; default to 4-bit
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
        """Bytes per parameter for the model's native dtype."""
        return BYTES_PER_PARAM.get(self.model.default_dtype, 2)

    @property
    def _base_model_bytes_per_param(self) -> float:
        """Effective bytes per parameter for base model weights.

        For quantized models (QLoRA, LoRA+quant), uses quantization bytes.
        Otherwise uses model's native dtype.
        """
        quant = self.config.quantization
        if quant != Quantization.NONE:
            return QUANTIZATION_BYTES[quant]
        return float(self._dtype_bytes)

    # ─────────────────────────────────────────────────────
    # LoRA parameter calculations
    # ─────────────────────────────────────────────────────

    def _count_lora_target_modules(self) -> int:
        """Count the number of linear layers targeted by LoRA per transformer layer.

        Returns:
            Number of target modules per layer.
        """
        target = self.config.lora_target
        if target == LoRATarget.ATTENTION:
            return 2  # q_proj, v_proj
        elif target == LoRATarget.ATTENTION_ALL:
            return 4  # q, k, v, o projections
        elif target == LoRATarget.ALL_LINEAR:
            return 7  # q, k, v, o + gate, up, down (MLP)
        return 2  # safe default

    def _compute_lora_params(self) -> int:
        """Calculate total number of trainable LoRA parameters.

        Each LoRA adapter has two low-rank matrices: A (d×r) and B (r×d)
        Total per module = 2 × hidden_size × rank

        Returns:
            Total trainable LoRA parameter count.
        """
        modules_per_layer = self._count_lora_target_modules()
        rank = self.config.lora_rank
        hidden = self.model.hidden_size

        # Each target module: A matrix (hidden × rank) + B matrix (rank × hidden)
        params_per_module = 2 * hidden * rank
        total = params_per_module * modules_per_layer * self.model.num_layers

        return total

    # ─────────────────────────────────────────────────────
    # Memory component calculations
    # ─────────────────────────────────────────────────────

    def _model_weights_bytes(self) -> float:
        """Calculate memory for base model weights.

        For quantized models, includes quantization overhead (block scaling constants).
        """
        base = self.model.parameters * self._base_model_bytes_per_param

        # Add quantization overhead if applicable
        quant = self.config.quantization
        if quant in QUANTIZATION_OVERHEAD_FRACTION:
            overhead_frac = QUANTIZATION_OVERHEAD_FRACTION[quant]
            base += self.model.parameters * self._base_model_bytes_per_param * overhead_frac

        return base

    def _trainable_params_bytes(self) -> float:
        """Calculate memory for trainable parameter storage.

        Full fine-tuning: 0 (already counted in model weights).
        LoRA/QLoRA: LoRA adapter weights in bf16/fp16.
        """
        if self.config.method == FineTuneMethod.FULL:
            return 0.0  # Trainable params are the model weights themselves

        lora_params = self._compute_lora_params()
        return lora_params * 2  # LoRA adapters stored in bf16 (2 bytes)

    def _gradient_bytes(self) -> float:
        """Calculate memory for gradient storage.

        Gradients are stored for trainable parameters only.
        Typically in fp16/bf16 (2 bytes per param).
        """
        trainable = self._get_trainable_param_count()
        return trainable * 2  # Gradients in bf16/fp16

    def _optimizer_state_bytes(self) -> float:
        """Calculate memory for optimizer states.

        Adam/AdamW: 8 bytes/param (two fp32 states: momentum + variance)
        SGD: 4 bytes/param (one fp32 state: momentum)
        8-bit Adam: 2 bytes/param (two int8 states)
        """
        trainable = self._get_trainable_param_count()
        bytes_per_param = OPTIMIZER_BYTES_PER_PARAM[self.config.optimizer]
        return trainable * bytes_per_param

    def _activation_bytes(self) -> float:
        """Calculate memory for activations during forward/backward pass.

        Activation memory depends on batch size, sequence length, hidden size,
        and whether gradient checkpointing is enabled.

        Formula:
            activations ≈ batch × seq_len × hidden_size × num_layers × bytes × factor
        Where factor accounts for intermediate computations and attention matrices.
        """
        batch = self.config.batch_size
        seq = self.config.seq_length
        hidden = self.model.hidden_size
        layers = self.model.num_layers
        dtype_bytes = 2  # Activations in fp16/bf16

        factor = (
            ACTIVATION_FACTOR_WITH_CHECKPOINTING
            if self.config.gradient_checkpointing
            else ACTIVATION_FACTOR_WITHOUT_CHECKPOINTING
        )

        return batch * seq * hidden * layers * dtype_bytes * factor

    def _get_trainable_param_count(self) -> int:
        """Get the number of trainable parameters based on method."""
        if self.config.method == FineTuneMethod.FULL:
            return self.model.parameters
        return self._compute_lora_params()

    # ─────────────────────────────────────────────────────
    # Main estimation
    # ─────────────────────────────────────────────────────

    def estimate(self) -> MemoryBreakdown:
        """Calculate complete VRAM breakdown for the training configuration.

        Returns:
            MemoryBreakdown with all components in GB and metadata.
        """
        model_weights = self._model_weights_bytes()
        trainable_mem = self._trainable_params_bytes()
        gradients = self._gradient_bytes()
        optimizer = self._optimizer_state_bytes()
        activations = self._activation_bytes()

        subtotal = model_weights + trainable_mem + gradients + optimizer + activations
        overhead = subtotal * CUDA_OVERHEAD_FRACTION
        total = subtotal + overhead

        trainable_params = self._get_trainable_param_count()
        trainable_pct = (trainable_params / self.model.parameters) * 100

        return MemoryBreakdown(
            model_weights_gb=round(model_weights * BYTES_TO_GB, 3),
            trainable_params_gb=round(trainable_mem * BYTES_TO_GB, 3),
            gradients_gb=round(gradients * BYTES_TO_GB, 3),
            optimizer_states_gb=round(optimizer * BYTES_TO_GB, 3),
            activations_gb=round(activations * BYTES_TO_GB, 3),
            overhead_gb=round(overhead * BYTES_TO_GB, 3),
            total_gb=round(total * BYTES_TO_GB, 3),
            total_params=self.model.parameters,
            trainable_params=trainable_params,
            trainable_percentage=round(trainable_pct, 4),
            method=self.config.method.value,
            quantization=self.config.quantization.value,
        )

    def check_gpu_fit(self, gpus: List[GPUSpec]) -> List[GPUFit]:
        """Check which GPUs can fit the estimated memory requirement.

        Args:
            gpus: List of GPU specifications to check against.

        Returns:
            List of GPUFit results sorted by utilization (best fit first).
        """
        mem = self.estimate()
        results = []

        for gpu in gpus:
            utilization = (mem.total_gb / gpu.vram_gb) * 100
            headroom = gpu.vram_gb - mem.total_gb

            results.append(
                GPUFit(
                    gpu_name=gpu.name,
                    vram_gb=gpu.vram_gb,
                    required_gb=mem.total_gb,
                    fits=mem.total_gb <= gpu.vram_gb,
                    utilization_percent=round(utilization, 1),
                    headroom_gb=round(headroom, 2),
                )
            )

        # Sort: fitting GPUs first (by utilization desc), then non-fitting
        results.sort(key=lambda g: (not g.fits, -g.headroom_gb))
        return results
