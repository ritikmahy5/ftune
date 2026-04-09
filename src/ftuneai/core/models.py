"""Data models for ftune configuration and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class FineTuneMethod(str, Enum):
    """Supported fine-tuning methods."""
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


class Quantization(str, Enum):
    """Quantization levels for base model weights."""
    NONE = "none"
    INT8 = "8bit"
    INT4 = "4bit"


class OptimizerType(str, Enum):
    """Supported optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAM_8BIT = "adam_8bit"
    ADAFACTOR = "adafactor"


class LoRATarget(str, Enum):
    """Which modules to apply LoRA adapters to."""
    ATTENTION = "attention"          # q_proj, v_proj only
    ATTENTION_ALL = "attention_all"  # q, k, v, o projections
    ALL_LINEAR = "all_linear"        # attention + MLP layers


class ShardingStrategy(str, Enum):
    """Multi-GPU sharding strategies."""
    NONE = "none"              # Single GPU or naive data parallel
    ZERO_1 = "zero_1"         # ZeRO Stage 1: shard optimizer states
    ZERO_2 = "zero_2"         # ZeRO Stage 2: shard optimizer + gradients
    ZERO_3 = "zero_3"         # ZeRO Stage 3: shard optimizer + gradients + parameters
    FSDP = "fsdp"             # PyTorch FSDP (similar to ZeRO-3)
    FSDP_SHARD_GRAD = "fsdp_shard_grad"  # FSDP shard grad+optimizer only (like ZeRO-2)


@dataclass
class ModelSpec:
    """Architecture specs for a single model."""
    name: str
    parameters: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_seq_length: int
    default_dtype: str = "bfloat16"
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_active_experts: Optional[int] = None


@dataclass
class GPUSpec:
    """Hardware specs for a single GPU."""
    name: str
    vram_gb: float
    fp16_tflops: float
    bf16_tflops: float
    fp32_tflops: float
    memory_bandwidth_gbps: float
    architecture: str = ""
    tdp_watts: int = 0


@dataclass
class TrainingConfig:
    """User-provided training configuration."""
    model: str = ""
    method: FineTuneMethod = FineTuneMethod.LORA
    quantization: Quantization = Quantization.NONE
    batch_size: int = 4
    seq_length: int = 2048
    gradient_checkpointing: bool = True
    optimizer: OptimizerType = OptimizerType.ADAMW
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target: LoRATarget = LoRATarget.ATTENTION
    # Advanced
    flash_attention: bool = False
    sharding: ShardingStrategy = ShardingStrategy.NONE
    num_gpus: int = 1


@dataclass
class MemoryBreakdown:
    """Detailed VRAM breakdown from memory estimation."""
    model_weights_gb: float
    trainable_params_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    overhead_gb: float
    total_gb: float
    total_params: int = 0
    trainable_params: int = 0
    trainable_percentage: float = 0.0
    method: str = ""
    quantization: str = ""
    sharding: str = "none"


@dataclass
class GPUFit:
    """Whether a training config fits on a specific GPU."""
    gpu_name: str
    vram_gb: float
    required_gb: float
    fits: bool
    utilization_percent: float = 0.0
    headroom_gb: float = 0.0


@dataclass
class TimeEstimate:
    """Training time estimation results."""
    total_hours: float
    hours_per_epoch: float
    total_steps: int
    steps_per_epoch: int
    total_tokens: int
    total_tflops: float
    gpu_name: str = ""
    num_gpus: int = 1
    mfu: float = 0.0
    dataset_size: int = 0
    epochs: int = 1


@dataclass
class CostEstimate:
    """Cost estimate for a single provider/GPU combination."""
    provider: str
    gpu: str
    gpu_count: int
    hourly_rate: float
    training_hours: float
    total_cost: float
    spot_hourly_rate: Optional[float] = None
    spot_total_cost: Optional[float] = None
    instance_type: str = ""
    region: str = ""


@dataclass
class CostComparison:
    """Cost comparison across all providers."""
    estimates: list  # List[CostEstimate]
    training_hours: float
    cheapest: Optional[str] = None
    best_value: Optional[str] = None


@dataclass
class CalibrationResult:
    """Result from a hardware calibration benchmark."""
    measured_mfu: float
    measured_memory_gb: float
    measured_throughput_tokens_per_sec: float
    hardware_multiplier: float  # actual_time / theoretical_time
    memory_multiplier: float    # actual_memory / estimated_memory
    gpu_name: str = ""
    steps_benchmarked: int = 10
    notes: List[str] = field(default_factory=list)


@dataclass
class BudgetRecommendation:
    """Configuration recommended by the budget optimizer."""
    method: str
    quantization: str
    lora_rank: int
    lora_target: str
    batch_size: int
    gradient_accumulation: int
    gradient_checkpointing: bool
    optimizer: str
    flash_attention: bool
    sharding: str
    # Results
    estimated_memory_gb: float
    estimated_hours: float
    estimated_cost: float
    gpu: str
    provider: str
    num_gpus: int
    fits: bool
    # Meta
    priority: str = ""  # "cost", "speed", "quality"
    notes: List[str] = field(default_factory=list)
