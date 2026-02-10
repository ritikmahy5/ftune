"""Tests for memory estimation engine.

Tests are validated against known real-world VRAM usage from
published benchmarks and community reports.
"""

import pytest
from ftune import Estimator, list_model_names, list_gpu_names, get_model, get_gpu


# ─────────────────────────────────────────────────────────
# Data loading tests
# ─────────────────────────────────────────────────────────


class TestDataLoading:
    """Tests for model and GPU database loading."""

    def test_models_load(self):
        models = list_model_names()
        assert len(models) > 10
        assert "meta-llama/Llama-3.1-8B" in models

    def test_gpus_load(self):
        gpus = list_gpu_names()
        assert len(gpus) > 5
        assert "A100-80GB" in gpus

    def test_get_model_exact(self):
        spec = get_model("meta-llama/Llama-3.1-8B")
        assert spec.parameters > 8_000_000_000
        assert spec.hidden_size == 4096

    def test_get_model_fuzzy(self):
        spec = get_model("Llama-3.1-8B")
        assert spec.name == "meta-llama/Llama-3.1-8B"

    def test_get_model_not_found(self):
        with pytest.raises(KeyError, match="not found"):
            get_model("nonexistent-model-xyz")

    def test_get_gpu_exact(self):
        spec = get_gpu("A100-80GB")
        assert spec.vram_gb == 80

    def test_get_gpu_fuzzy(self):
        spec = get_gpu("a100-80gb")
        assert spec.vram_gb == 80

    def test_get_gpu_not_found(self):
        with pytest.raises(KeyError, match="not found"):
            get_gpu("nonexistent-gpu-xyz")


# ─────────────────────────────────────────────────────────
# Memory estimation: QLoRA
# ─────────────────────────────────────────────────────────


class TestQLoRAMemory:
    """QLoRA memory estimation tests."""

    def test_llama_8b_qlora_fits_on_24gb(self, llama_8b_qlora):
        """Llama 8B QLoRA 4-bit should fit on 24GB GPUs."""
        mem = llama_8b_qlora.estimate_memory()
        assert mem.total_gb < 24, f"Expected <24GB, got {mem.total_gb:.2f}GB"

    def test_llama_8b_qlora_realistic_range(self, llama_8b_qlora):
        """QLoRA 8B should use roughly 10-20GB (known from practice)."""
        mem = llama_8b_qlora.estimate_memory()
        assert 5 < mem.total_gb < 24, f"Expected 5-24GB, got {mem.total_gb:.2f}GB"

    def test_qlora_base_model_is_quantized(self, llama_8b_qlora):
        """4-bit quantized 8B model should be ~4-5GB for weights."""
        mem = llama_8b_qlora.estimate_memory()
        assert 3 < mem.model_weights_gb < 6, (
            f"4-bit 8B model weights should be ~4-5GB, got {mem.model_weights_gb:.2f}GB"
        )

    def test_qlora_trainable_percentage_small(self, llama_8b_qlora):
        """QLoRA should have <1% trainable parameters."""
        mem = llama_8b_qlora.estimate_memory()
        assert mem.trainable_percentage < 1.0, (
            f"Expected <1% trainable, got {mem.trainable_percentage:.4f}%"
        )

    def test_qlora_auto_sets_quantization(self):
        """QLoRA method should auto-set 4bit quantization if none specified."""
        est = Estimator(model="meta-llama/Llama-3.1-8B", method="qlora")
        mem = est.estimate_memory()
        assert mem.quantization == "4bit"

    def test_qlora_8bit(self):
        """8-bit QLoRA should use more memory than 4-bit."""
        est_4bit = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        est_8bit = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="8bit", batch_size=4, seq_length=2048,
        )
        mem_4 = est_4bit.estimate_memory()
        mem_8 = est_8bit.estimate_memory()
        assert mem_8.model_weights_gb > mem_4.model_weights_gb

    def test_llama_70b_qlora_fits_on_a100_80gb(self):
        """70B QLoRA 4-bit should fit on A100 80GB with small batch."""
        est = Estimator(
            model="meta-llama/Llama-3.1-70B", method="qlora",
            quantization="4bit", lora_rank=16, batch_size=1, seq_length=2048,
        )
        mem = est.estimate_memory()
        assert mem.total_gb < 80, f"70B QLoRA should fit on 80GB, got {mem.total_gb:.2f}GB"


# ─────────────────────────────────────────────────────────
# Memory estimation: LoRA (no quantization)
# ─────────────────────────────────────────────────────────


class TestLoRAMemory:
    """LoRA memory estimation tests."""

    def test_llama_8b_lora_needs_more_than_qlora(self, llama_8b_lora, llama_8b_qlora):
        """LoRA (fp16 base) should need more VRAM than QLoRA (4-bit base)."""
        mem_lora = llama_8b_lora.estimate_memory()
        mem_qlora = llama_8b_qlora.estimate_memory()
        assert mem_lora.total_gb > mem_qlora.total_gb

    def test_lora_base_model_fp16(self, llama_8b_lora):
        """LoRA base model (bf16) should be ~16GB for 8B params."""
        mem = llama_8b_lora.estimate_memory()
        assert 14 < mem.model_weights_gb < 18, (
            f"bf16 8B model should be ~16GB, got {mem.model_weights_gb:.2f}GB"
        )

    def test_lora_rank_affects_trainable_params(self):
        """Higher LoRA rank should mean more trainable parameters."""
        est_r8 = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora", lora_rank=8,
        )
        est_r64 = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora", lora_rank=64,
        )
        mem_r8 = est_r8.estimate_memory()
        mem_r64 = est_r64.estimate_memory()
        assert mem_r64.trainable_params > mem_r8.trainable_params
        assert mem_r64.optimizer_states_gb > mem_r8.optimizer_states_gb

    def test_lora_target_all_linear_more_params(self):
        """Targeting all linear layers should have more params than attention only."""
        est_attn = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora",
            lora_rank=16, lora_target="attention",
        )
        est_all = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora",
            lora_rank=16, lora_target="all_linear",
        )
        mem_attn = est_attn.estimate_memory()
        mem_all = est_all.estimate_memory()
        assert mem_all.trainable_params > mem_attn.trainable_params


# ─────────────────────────────────────────────────────────
# Memory estimation: Full Fine-Tuning
# ─────────────────────────────────────────────────────────


class TestFullFineTuneMemory:
    """Full fine-tuning memory estimation tests."""

    def test_llama_8b_full_needs_a100(self, llama_8b_full):
        """Full fine-tune 8B model should need significant VRAM (>40GB)."""
        mem = llama_8b_full.estimate_memory()
        assert mem.total_gb > 40, f"Full 8B fine-tune should need >40GB, got {mem.total_gb:.2f}GB"

    def test_llama_70b_full_exceeds_single_gpu(self, llama_70b_full):
        """Full 70B fine-tune should NOT fit on any single GPU."""
        mem = llama_70b_full.estimate_memory()
        assert mem.total_gb > 80, f"Full 70B should need >80GB, got {mem.total_gb:.2f}GB"

    def test_full_finetune_all_params_trainable(self, llama_8b_full):
        """Full fine-tuning should show 100% trainable params."""
        mem = llama_8b_full.estimate_memory()
        assert mem.trainable_percentage == 100.0

    def test_full_more_memory_than_lora(self, llama_8b_full, llama_8b_lora):
        """Full fine-tuning should use more memory than LoRA."""
        mem_full = llama_8b_full.estimate_memory()
        mem_lora = llama_8b_lora.estimate_memory()
        assert mem_full.total_gb > mem_lora.total_gb

    def test_full_quantization_raises_error(self):
        """Full fine-tuning with quantization should raise ValueError."""
        with pytest.raises(ValueError, match="does not support quantization"):
            Estimator(
                model="meta-llama/Llama-3.1-8B",
                method="full",
                quantization="4bit",
            )


# ─────────────────────────────────────────────────────────
# Gradient checkpointing
# ─────────────────────────────────────────────────────────


class TestGradientCheckpointing:
    """Tests for gradient checkpointing effect on memory."""

    def test_checkpointing_reduces_activations(self):
        """Gradient checkpointing should significantly reduce activation memory."""
        est_on = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora",
            gradient_checkpointing=True, batch_size=4, seq_length=2048,
        )
        est_off = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora",
            gradient_checkpointing=False, batch_size=4, seq_length=2048,
        )
        mem_on = est_on.estimate_memory()
        mem_off = est_off.estimate_memory()

        assert mem_on.activations_gb < mem_off.activations_gb
        # Should be roughly 5x reduction
        ratio = mem_off.activations_gb / mem_on.activations_gb
        assert ratio > 3, f"Expected >3x reduction, got {ratio:.1f}x"

    def test_checkpointing_reduces_total(self):
        """Total memory should be lower with checkpointing."""
        est_on = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora",
            gradient_checkpointing=True, batch_size=4, seq_length=2048,
        )
        est_off = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora",
            gradient_checkpointing=False, batch_size=4, seq_length=2048,
        )
        assert est_on.estimate_memory().total_gb < est_off.estimate_memory().total_gb


# ─────────────────────────────────────────────────────────
# Batch size and sequence length scaling
# ─────────────────────────────────────────────────────────


class TestScaling:
    """Tests for memory scaling with batch size and sequence length."""

    def test_larger_batch_more_memory(self):
        """Doubling batch size should increase activation memory."""
        est_bs1 = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=1, seq_length=2048,
        )
        est_bs8 = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=8, seq_length=2048,
        )
        mem_1 = est_bs1.estimate_memory()
        mem_8 = est_bs8.estimate_memory()
        assert mem_8.activations_gb > mem_1.activations_gb
        assert mem_8.total_gb > mem_1.total_gb

    def test_longer_seq_more_memory(self):
        """Longer sequence length should increase memory."""
        est_short = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=512,
        )
        est_long = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=4096,
        )
        mem_short = est_short.estimate_memory()
        mem_long = est_long.estimate_memory()
        assert mem_long.activations_gb > mem_short.activations_gb

    def test_seq_length_exceeds_model_max(self):
        """Requesting seq_length beyond model max should raise error."""
        with pytest.raises(ValueError, match="exceeds model's max_seq_length"):
            Estimator(
                model="microsoft/phi-3-mini-4k-instruct",
                method="lora",
                seq_length=999_999,
            )


# ─────────────────────────────────────────────────────────
# GPU fit checks
# ─────────────────────────────────────────────────────────


class TestGPUFit:
    """Tests for GPU compatibility checking."""

    def test_qlora_8b_fits_on_rtx4090(self, llama_8b_qlora):
        """QLoRA 8B should fit on RTX 4090 (24GB)."""
        fits = llama_8b_qlora.check_gpu_fit(gpu_names=["RTX-4090-24GB"])
        assert len(fits) == 1
        assert fits[0].fits is True

    def test_qlora_8b_fits_on_a100(self, llama_8b_qlora):
        """QLoRA 8B should fit on A100 80GB with plenty of headroom."""
        fits = llama_8b_qlora.check_gpu_fit(gpu_names=["A100-80GB"])
        assert fits[0].fits is True
        assert fits[0].headroom_gb > 50  # Lots of room

    def test_full_70b_too_large_for_a100(self, llama_70b_full):
        """Full 70B should not fit on a single A100-80GB."""
        fits = llama_70b_full.check_gpu_fit(gpu_names=["A100-80GB"])
        assert fits[0].fits is False

    def test_check_all_gpus(self, llama_8b_qlora):
        """Check against all GPUs should return results for each."""
        fits = llama_8b_qlora.check_gpu_fit()
        gpu_count = len(list_gpu_names())
        assert len(fits) == gpu_count

    def test_fit_results_sorted(self, llama_8b_qlora):
        """Fitting GPUs should come before non-fitting ones."""
        fits = llama_8b_qlora.check_gpu_fit()
        fitting = [f for f in fits if f.fits]
        not_fitting = [f for f in fits if not f.fits]
        if fitting and not_fitting:
            # All fitting GPUs should appear before non-fitting
            last_fit_idx = max(fits.index(f) for f in fitting)
            first_no_fit_idx = min(fits.index(f) for f in not_fitting)
            assert last_fit_idx < first_no_fit_idx


# ─────────────────────────────────────────────────────────
# Optimizer variations
# ─────────────────────────────────────────────────────────


class TestOptimizers:
    """Tests for different optimizer memory footprints."""

    def test_adam_8bit_less_than_adamw(self):
        """8-bit Adam should use less optimizer memory than AdamW."""
        est_adamw = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora",
            lora_rank=16, optimizer="adamw",
        )
        est_8bit = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora",
            lora_rank=16, optimizer="adam_8bit",
        )
        mem_adamw = est_adamw.estimate_memory()
        mem_8bit = est_8bit.estimate_memory()
        assert mem_8bit.optimizer_states_gb < mem_adamw.optimizer_states_gb

    def test_sgd_less_than_adam(self):
        """SGD should use less optimizer memory than Adam."""
        est_adam = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora",
            lora_rank=16, optimizer="adamw",
        )
        est_sgd = Estimator(
            model="meta-llama/Llama-3.1-8B", method="lora",
            lora_rank=16, optimizer="sgd",
        )
        mem_adam = est_adam.estimate_memory()
        mem_sgd = est_sgd.estimate_memory()
        assert mem_sgd.optimizer_states_gb < mem_adam.optimizer_states_gb


# ─────────────────────────────────────────────────────────
# Summary / display
# ─────────────────────────────────────────────────────────


class TestSummary:
    """Tests for summary output."""

    def test_summary_runs(self, llama_8b_qlora):
        """Summary should return a non-empty string."""
        summary = llama_8b_qlora.summary()
        assert isinstance(summary, str)
        assert len(summary) > 100

    def test_summary_contains_model_name(self, llama_8b_qlora):
        """Summary should mention the model name."""
        summary = llama_8b_qlora.summary()
        assert "Llama-3.1-8B" in summary

    def test_summary_contains_total(self, llama_8b_qlora):
        """Summary should contain TOTAL line."""
        summary = llama_8b_qlora.summary()
        assert "TOTAL" in summary


# ─────────────────────────────────────────────────────────
# Cross-model sanity checks
# ─────────────────────────────────────────────────────────


class TestCrossModel:
    """Sanity checks across different models."""

    def test_larger_model_needs_more_memory(self):
        """70B model should always need more memory than 8B."""
        est_8b = Estimator(model="meta-llama/Llama-3.1-8B", method="qlora", quantization="4bit")
        est_70b = Estimator(model="meta-llama/Llama-3.1-70B", method="qlora", quantization="4bit")
        assert est_70b.estimate_memory().total_gb > est_8b.estimate_memory().total_gb

    def test_mistral_7b_similar_to_llama_8b(self):
        """Mistral 7B and Llama 8B should have similar VRAM needs."""
        est_llama = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        est_mistral = Estimator(
            model="mistralai/Mistral-7B-v0.3", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        mem_llama = est_llama.estimate_memory().total_gb
        mem_mistral = est_mistral.estimate_memory().total_gb
        # Should be within 50% of each other
        ratio = max(mem_llama, mem_mistral) / min(mem_llama, mem_mistral)
        assert ratio < 1.5, f"Expected similar memory, got ratio {ratio:.2f}"

    def test_phi3_mini_fits_on_t4(self):
        """Phi-3 Mini (3.8B) QLoRA should fit on T4 16GB."""
        est = Estimator(
            model="microsoft/phi-3-mini-4k-instruct",
            method="qlora", quantization="4bit",
            batch_size=2, seq_length=2048,
        )
        fits = est.check_gpu_fit(gpu_names=["T4-16GB"])
        assert fits[0].fits is True
