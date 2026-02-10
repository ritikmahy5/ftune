"""Shared test fixtures."""

import pytest
from ftune import Estimator


@pytest.fixture
def llama_8b_qlora():
    """Llama 3.1 8B with QLoRA 4-bit, typical config."""
    return Estimator(
        model="meta-llama/Llama-3.1-8B",
        method="qlora",
        quantization="4bit",
        lora_rank=16,
        batch_size=4,
        seq_length=2048,
    )


@pytest.fixture
def llama_8b_lora():
    """Llama 3.1 8B with LoRA, no quantization."""
    return Estimator(
        model="meta-llama/Llama-3.1-8B",
        method="lora",
        lora_rank=16,
        batch_size=4,
        seq_length=2048,
    )


@pytest.fixture
def llama_8b_full():
    """Llama 3.1 8B full fine-tuning."""
    return Estimator(
        model="meta-llama/Llama-3.1-8B",
        method="full",
        batch_size=1,
        seq_length=2048,
    )


@pytest.fixture
def llama_70b_full():
    """Llama 3.1 70B full fine-tuning."""
    return Estimator(
        model="meta-llama/Llama-3.1-70B",
        method="full",
        batch_size=1,
        seq_length=2048,
    )
