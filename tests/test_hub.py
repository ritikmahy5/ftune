"""Tests for HuggingFace Hub integration.

Uses mocked HTTP responses to avoid network dependency.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from ftune.hub import (
    _extract_field,
    _estimate_param_count,
    _detect_dtype,
    fetch_model_config,
    resolve_model_from_hub,
)


# ─────────────────────────────────────────────────────────
# Sample configs for testing
# ─────────────────────────────────────────────────────────

LLAMA_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
}

MIXTRAL_MOE_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
}

GPT2_CONFIG = {
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "vocab_size": 50257,
    "n_positions": 1024,
}


def _mock_urlopen(config_dict):
    """Create a mock for urllib.request.urlopen that returns config as JSON."""
    resp = MagicMock()
    data = json.dumps(config_dict).encode("utf-8")
    resp.read.return_value = data
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ─────────────────────────────────────────────────────────
# Field extraction tests
# ─────────────────────────────────────────────────────────


class TestExtractField:
    """Tests for _extract_field with alias resolution."""

    def test_standard_key(self):
        assert _extract_field(LLAMA_CONFIG, "hidden_size") == 4096

    def test_alias_n_embd(self):
        """GPT-2 uses n_embd instead of hidden_size."""
        assert _extract_field(GPT2_CONFIG, "hidden_size") == 768

    def test_alias_n_layer(self):
        assert _extract_field(GPT2_CONFIG, "num_layers") == 12

    def test_alias_n_positions(self):
        assert _extract_field(GPT2_CONFIG, "max_seq_length") == 1024

    def test_missing_field_returns_none(self):
        assert _extract_field({}, "hidden_size") is None

    def test_kv_heads_fallback_to_attn_heads(self):
        """When no specific KV head key, falls back to num_attention_heads."""
        config = {"num_attention_heads": 32}
        assert _extract_field(config, "num_kv_heads") == 32


# ─────────────────────────────────────────────────────────
# Parameter estimation tests
# ─────────────────────────────────────────────────────────


class TestEstimateParamCount:
    """Tests for transformer parameter count estimation."""

    def test_llama_8b_estimate(self):
        """Estimated params for Llama config should be in 7-9B range."""
        params = _estimate_param_count(LLAMA_CONFIG)
        assert 7e9 < params < 10e9, f"Expected ~8B params, got {params:,}"

    def test_gpt2_small_estimate(self):
        """GPT-2 small should be ~100-200M params."""
        params = _estimate_param_count(GPT2_CONFIG)
        assert 80e6 < params < 250e6, f"Expected ~124M params, got {params:,}"

    def test_larger_hidden_size_more_params(self):
        """Larger hidden size should mean more parameters."""
        small = _estimate_param_count({"hidden_size": 2048, "num_hidden_layers": 16, "vocab_size": 32000})
        large = _estimate_param_count({"hidden_size": 8192, "num_hidden_layers": 16, "vocab_size": 32000})
        assert large > small


# ─────────────────────────────────────────────────────────
# dtype detection tests
# ─────────────────────────────────────────────────────────


class TestDetectDtype:
    def test_bfloat16(self):
        assert _detect_dtype({"torch_dtype": "bfloat16"}) == "bfloat16"

    def test_float16(self):
        assert _detect_dtype({"torch_dtype": "float16"}) == "float16"

    def test_default_is_bfloat16(self):
        assert _detect_dtype({}) == "bfloat16"

    def test_shorthand_bf16(self):
        assert _detect_dtype({"torch_dtype": "bf16"}) == "bfloat16"


# ─────────────────────────────────────────────────────────
# MoE detection tests
# ─────────────────────────────────────────────────────────


class TestMoEDetection:
    """Tests for Mixture of Experts model detection."""

    @patch("ftune.hub.urlopen")
    def test_moe_model_detected(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(MIXTRAL_MOE_CONFIG)
        spec = resolve_model_from_hub("mistralai/Mixtral-8x7B-v0.1")
        assert spec.is_moe is True
        assert spec.num_experts == 8
        assert spec.num_active_experts == 2

    @patch("ftune.hub.urlopen")
    def test_non_moe_model(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(LLAMA_CONFIG)
        spec = resolve_model_from_hub("meta-llama/Llama-3.1-8B")
        assert spec.is_moe is False


# ─────────────────────────────────────────────────────────
# Model name validation
# ─────────────────────────────────────────────────────────


class TestModelNameValidation:
    def test_valid_model_name(self):
        """Valid org/model format should not raise."""
        # Just test the validation, not the network call
        with patch("ftune.hub.urlopen") as mock:
            mock.return_value = _mock_urlopen(LLAMA_CONFIG)
            config = fetch_model_config("meta-llama/Llama-3.1-8B")
            assert isinstance(config, dict)

    def test_invalid_model_name_no_slash(self):
        """Model name without slash should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid model name"):
            fetch_model_config("just-a-model-name")

    def test_invalid_model_name_path_traversal(self):
        """Model name with path traversal should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid model name"):
            fetch_model_config("../../etc/passwd")


# ─────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────


class TestErrorHandling:
    @patch("ftune.hub.urlopen")
    def test_network_error_raises_value_error(self, mock_urlopen):
        """Network failure should raise ValueError with helpful message."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")
        with pytest.raises(ValueError, match="Could not fetch config"):
            fetch_model_config("org/nonexistent-model")

    @patch("ftune.hub.urlopen")
    def test_missing_architecture_fields(self, mock_urlopen):
        """Config without required fields should raise ValueError."""
        mock_urlopen.return_value = _mock_urlopen({"some_random_key": 42})
        with pytest.raises(ValueError, match="Could not extract"):
            resolve_model_from_hub("org/bad-model")
