"""Tests for validation module."""

from ftune import Estimator
from ftune.validation import Validator, ActualMetrics, ValidationResult


class TestValidation:
    """Validation comparison tests."""

    def test_memory_comparison(self):
        """Should compute memory error correctly."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        actual = ActualMetrics(peak_memory_gb=11.0, gpu_name="A100-80GB")
        result = Validator.compare(est, actual)

        assert result.estimated_memory_gb is not None
        assert result.actual_memory_gb == 11.0
        assert result.memory_error_pct is not None
        assert result.memory_error_gb is not None

    def test_time_comparison(self):
        """Should compute time error when all info is available."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        actual = ActualMetrics(
            peak_memory_gb=11.0,
            training_time_hours=5.0,
            gpu_name="A100-80GB",
            dataset_size=50000,
            epochs=3,
        )
        result = Validator.compare(est, actual)

        assert result.estimated_time_hours is not None
        assert result.actual_time_hours == 5.0
        assert result.time_error_pct is not None

    def test_overestimate_detected(self):
        """Overestimation should produce positive error."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        # Set actual much lower than estimate
        actual = ActualMetrics(peak_memory_gb=5.0)
        result = Validator.compare(est, actual)
        assert result.memory_error_pct > 0  # Overestimate

    def test_underestimate_detected(self):
        """Underestimation should produce negative error."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        # Set actual much higher than estimate
        actual = ActualMetrics(peak_memory_gb=50.0)
        result = Validator.compare(est, actual)
        assert result.memory_error_pct < 0  # Underestimate

    def test_accuracy_grade(self):
        """Should assign appropriate accuracy grades."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        # Close estimate = good grade
        mem = est.estimate_memory()
        actual = ActualMetrics(peak_memory_gb=mem.total_gb * 1.05)  # 5% off
        result = Validator.compare(est, actual)
        assert "Excellent" in result.accuracy_grade or "Good" in result.accuracy_grade

    def test_format_report(self):
        """Report should be a non-empty string with key sections."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        actual = ActualMetrics(
            peak_memory_gb=11.0,
            training_time_hours=5.0,
            gpu_name="A100-80GB",
            dataset_size=50000,
            epochs=3,
        )
        result = Validator.compare(est, actual)
        report = Validator.format_report(result)

        assert len(report) > 100
        assert "Memory" in report
        assert "Accuracy" in report

    def test_partial_metrics(self):
        """Should handle partial metrics gracefully."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        # Only memory, no time or cost
        actual = ActualMetrics(peak_memory_gb=10.0)
        result = Validator.compare(est, actual)

        assert result.memory_error_pct is not None
        assert result.time_error_pct is None  # No time data provided
        assert result.cost_error_pct is None  # No cost data provided

    def test_no_metrics(self):
        """Should handle empty metrics without crashing."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        actual = ActualMetrics()
        result = Validator.compare(est, actual)
        assert result.accuracy_grade == "N/A"
