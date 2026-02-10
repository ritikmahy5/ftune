"""Tests for training time estimation engine."""

from ftune import Estimator


class TestTimeEstimation:
    """Training time estimation tests."""

    def test_basic_time_estimate(self):
        """Should return a valid time estimate."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        time = est.estimate_time(
            gpu="A100-80GB", dataset_size=50000, epochs=3,
        )
        assert time.total_hours > 0
        assert time.hours_per_epoch > 0
        assert time.total_steps > 0
        assert time.gpu_name == "A100-80GB"
        assert time.epochs == 3

    def test_h100_faster_than_a100(self):
        """H100 should be faster than A100 for same workload."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        time_a100 = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3)
        time_h100 = est.estimate_time(gpu="H100-80GB", dataset_size=50000, epochs=3)
        assert time_h100.total_hours < time_a100.total_hours

    def test_more_data_more_time(self):
        """Larger dataset should take more time."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        time_small = est.estimate_time(gpu="A100-80GB", dataset_size=10000, epochs=1)
        time_large = est.estimate_time(gpu="A100-80GB", dataset_size=100000, epochs=1)
        assert time_large.total_hours > time_small.total_hours

    def test_more_epochs_more_time(self):
        """More epochs should take proportionally more time."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        time_1 = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=1)
        time_3 = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3)
        # Should be roughly 3x (not exact due to rounding)
        ratio = time_3.total_hours / time_1.total_hours
        assert 2.5 < ratio < 3.5

    def test_multi_gpu_faster(self):
        """Multi-GPU should be faster than single GPU."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        time_1gpu = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3, num_gpus=1)
        time_4gpu = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3, num_gpus=4)
        assert time_4gpu.total_hours < time_1gpu.total_hours

    def test_multi_gpu_not_perfect_scaling(self):
        """4 GPUs should NOT be exactly 4x faster (communication overhead)."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        time_1 = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3, num_gpus=1)
        time_4 = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3, num_gpus=4)
        speedup = time_1.total_hours / time_4.total_hours
        # Should be less than 4x due to overhead
        assert speedup < 4.0
        # But still significant
        assert speedup > 2.0

    def test_full_finetune_slower_than_qlora(self):
        """Full fine-tuning should take longer than QLoRA (more FLOPs)."""
        est_qlora = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        est_full = Estimator(
            model="meta-llama/Llama-3.1-8B", method="full",
            batch_size=4, seq_length=2048,
        )
        time_qlora = est_qlora.estimate_time(gpu="H100-80GB", dataset_size=50000, epochs=3)
        time_full = est_full.estimate_time(gpu="H100-80GB", dataset_size=50000, epochs=3)
        assert time_full.total_hours > time_qlora.total_hours

    def test_70b_slower_than_8b(self):
        """70B model should take much longer than 8B."""
        est_8b = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=1, seq_length=2048,
        )
        est_70b = Estimator(
            model="meta-llama/Llama-3.1-70B", method="qlora",
            quantization="4bit", batch_size=1, seq_length=2048,
        )
        time_8b = est_8b.estimate_time(gpu="A100-80GB", dataset_size=10000, epochs=1)
        time_70b = est_70b.estimate_time(gpu="A100-80GB", dataset_size=10000, epochs=1)
        assert time_70b.total_hours > time_8b.total_hours * 5

    def test_time_all_gpus(self):
        """Should return time estimates for compatible GPUs only."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        times = est.estimate_time_all_gpus(dataset_size=50000, epochs=3)
        assert len(times) > 0
        # Should be sorted fastest first
        for i in range(len(times) - 1):
            assert times[i].total_hours <= times[i + 1].total_hours

    def test_token_count(self):
        """Token count should match dataset_size × seq_length × epochs."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        time = est.estimate_time(gpu="A100-80GB", dataset_size=1000, epochs=2)
        assert time.total_tokens == 1000 * 2048 * 2

    def test_steps_calculation(self):
        """Steps per epoch should be dataset_size / batch_size."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        time = est.estimate_time(gpu="A100-80GB", dataset_size=1000, epochs=1)
        assert time.steps_per_epoch == 250  # 1000 / 4

    def test_reasonable_time_for_typical_setup(self):
        """Typical QLoRA setup should be in reasonable range (not days)."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", lora_rank=16, batch_size=4, seq_length=2048,
        )
        time = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3)
        # Should be hours, not days or minutes
        assert 0.5 < time.total_hours < 50, f"Expected 0.5-50 hrs, got {time.total_hours}"
