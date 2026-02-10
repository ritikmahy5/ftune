"""HuggingFace Hub integration for auto-detecting model specs.

Fetches model architecture details directly from HuggingFace Hub,
enabling ftune to work with ANY model — not just those in the
bundled database.
"""

from __future__ import annotations

import json
import re
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from ftune.core.models import ModelSpec


# HuggingFace Hub API base URL
HF_API_BASE = "https://huggingface.co"

# Mapping from HF config keys to our ModelSpec fields
# Different model architectures use different key names
_PARAM_ALIASES = {
    "hidden_size": ["hidden_size", "d_model", "n_embd", "dim"],
    "num_layers": ["num_hidden_layers", "n_layer", "num_layers", "n_layers"],
    "num_attention_heads": ["num_attention_heads", "n_head", "num_heads", "n_heads"],
    "num_kv_heads": [
        "num_key_value_heads", "num_kv_heads", "n_head_kv",
        "num_attention_heads",  # fallback: MHA (kv_heads == attn_heads)
    ],
    "intermediate_size": ["intermediate_size", "ffn_dim", "n_inner", "ffn_hidden_size"],
    "vocab_size": ["vocab_size"],
    "max_seq_length": [
        "max_position_embeddings", "max_seq_len", "max_sequence_length",
        "n_positions", "seq_length", "sliding_window",
    ],
}

# Known model parameter counts (billions) when not easily computable
# Used as fallback when config doesn't have explicit param count
_KNOWN_PARAM_COUNTS = {
    "llama": {4096: 8e9, 8192: 70e9, 16384: 405e9},      # hidden_size → approx params
    "mistral": {4096: 7.2e9, 8192: 46.7e9},
    "qwen": {3584: 7.6e9, 8192: 72.7e9},
    "gemma": {3584: 9.2e9, 4608: 27.2e9, 2048: 2.6e9},
    "phi": {3072: 3.8e9, 5120: 14e9},
}


def _fetch_json(url: str, timeout: int = 10) -> dict:
    """Fetch JSON from a URL with timeout."""
    req = Request(url, headers={"User-Agent": "ftune/0.1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError) as e:
        raise ConnectionError(f"Failed to fetch {url}: {e}") from e


def _extract_field(config: dict, field_name: str) -> Optional[int]:
    """Extract a field from HF config using known aliases."""
    aliases = _PARAM_ALIASES.get(field_name, [field_name])
    for alias in aliases:
        if alias in config:
            val = config[alias]
            if isinstance(val, (int, float)):
                return int(val)
    return None


def _estimate_param_count(config: dict) -> int:
    """Estimate total parameter count from model architecture.

    Uses the transformer parameter formula:
        params ≈ vocab_embed + num_layers × (attention + ffn + layernorm)

    Where:
        attention ≈ 4 × hidden² (Q, K, V, O projections)
        ffn ≈ 3 × hidden × intermediate (gate, up, down for SwiGLU)
              or 2 × hidden × intermediate (for standard FFN)
        layernorm ≈ 2 × hidden (per layer)
        vocab_embed ≈ vocab_size × hidden
    """
    hidden = _extract_field(config, "hidden_size") or 4096
    layers = _extract_field(config, "num_layers") or 32
    intermediate = _extract_field(config, "intermediate_size") or (hidden * 4)
    vocab = _extract_field(config, "vocab_size") or 32000
    num_heads = _extract_field(config, "num_attention_heads") or 32
    num_kv = _extract_field(config, "num_kv_heads") or num_heads

    # Embedding layer
    embed_params = vocab * hidden

    # Per-layer attention: Q, K, V, O projections
    head_dim = hidden // num_heads
    q_params = hidden * (num_heads * head_dim)      # Q projection
    k_params = hidden * (num_kv * head_dim)          # K projection
    v_params = hidden * (num_kv * head_dim)          # V projection
    o_params = (num_heads * head_dim) * hidden       # Output projection
    attn_params = q_params + k_params + v_params + o_params

    # Per-layer FFN (assume SwiGLU: gate + up + down)
    # Check if model uses gated FFN
    has_gate = config.get("hidden_act", "") in ("silu", "swiglu", "gelu_new")
    if has_gate or intermediate > hidden * 3:
        ffn_params = 3 * hidden * intermediate  # gate, up, down
    else:
        ffn_params = 2 * hidden * intermediate  # standard FFN

    # Layer norms (2 per layer: attn + ffn)
    norm_params = 4 * hidden  # 2 layernorms × 2 (weight + bias)

    # Total
    per_layer = attn_params + ffn_params + norm_params
    total = embed_params + (layers * per_layer) + (2 * hidden)  # final norm

    return int(total)


def _detect_dtype(config: dict) -> str:
    """Detect the default dtype from config."""
    dtype = config.get("torch_dtype", "bfloat16")
    dtype_map = {
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    return dtype_map.get(str(dtype).lower(), "bfloat16")


def fetch_model_config(model_name: str) -> dict:
    """Fetch the config.json for a model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model ID (e.g. 'meta-llama/Llama-3.1-8B').

    Returns:
        Raw config dictionary.

    Raises:
        ConnectionError: If the model can't be fetched.
        ValueError: If the model doesn't exist or config is missing.
    """
    url = f"{HF_API_BASE}/{model_name}/resolve/main/config.json"
    try:
        config = _fetch_json(url)
    except ConnectionError:
        raise ValueError(
            f"Could not fetch config for '{model_name}'. "
            f"Check that the model exists on HuggingFace Hub and you have internet access."
        )
    return config


def resolve_model_from_hub(model_name: str) -> ModelSpec:
    """Fetch model specs from HuggingFace Hub and create a ModelSpec.

    This enables ftune to work with ANY model on HuggingFace, not just
    those in the bundled database.

    Args:
        model_name: HuggingFace model ID (e.g. 'meta-llama/Llama-3.1-8B').

    Returns:
        ModelSpec with architecture details.

    Example:
        >>> spec = resolve_model_from_hub("mistralai/Mistral-7B-v0.3")
        >>> print(f"{spec.name}: {spec.parameters:,} params")
    """
    config = fetch_model_config(model_name)

    hidden_size = _extract_field(config, "hidden_size")
    num_layers = _extract_field(config, "num_layers")
    num_attention_heads = _extract_field(config, "num_attention_heads")
    num_kv_heads = _extract_field(config, "num_kv_heads") or num_attention_heads
    intermediate_size = _extract_field(config, "intermediate_size")
    vocab_size = _extract_field(config, "vocab_size")
    max_seq_length = _extract_field(config, "max_seq_length")

    if not all([hidden_size, num_layers, num_attention_heads]):
        raise ValueError(
            f"Could not extract required architecture fields from '{model_name}' config. "
            f"Keys found: {list(config.keys())}"
        )

    # Estimate parameters
    parameters = _estimate_param_count(config)

    # Check for MoE
    num_experts = config.get("num_local_experts") or config.get("num_experts")
    num_active = config.get("num_experts_per_tok") or config.get("num_active_experts")

    return ModelSpec(
        name=model_name,
        parameters=parameters,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads or num_attention_heads,
        intermediate_size=intermediate_size or (hidden_size * 4),
        vocab_size=vocab_size or 32000,
        max_seq_length=max_seq_length or 4096,
        default_dtype=_detect_dtype(config),
        is_moe=num_experts is not None,
        num_experts=num_experts,
        num_active_experts=num_active,
    )
