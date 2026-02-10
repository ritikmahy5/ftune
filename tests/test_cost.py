"""Tests for cloud cost estimation engine."""

from ftune import Estimator
from ftune.core.cost import CostEstimator, list_providers, get_provider_display_name


class TestCostDataLoading:
    """Tests for pricing database loading."""

    def test_providers_load(self):
        providers = list_providers()
        assert len(providers) >= 7
        assert "lambda_labs" in providers
        assert "runpod" in providers
        assert "aws" in providers

    def test_provider_display_names(self):
        assert get_provider_display_name("lambda_labs") == "Lambda Labs"
        assert get_provider_display_name("aws") == "AWS"
        assert get_provider_display_name("gcp") == "Google Cloud"


class TestCostEstimation:
    """Cost estimation tests."""

    def test_basic_cost_estimate(self):
        """Should return costs for A100 across multiple providers."""
        ce = CostEstimator()
        estimates = ce.estimate_for_gpu("A100-80GB", training_hours=5.0)
        assert len(estimates) > 0
        for e in estimates:
            assert e.total_cost > 0
            assert e.training_hours == 5.0
            assert e.gpu == "A100-80GB"

    def test_costs_sorted_by_price(self):
        """Estimates should be sorted cheapest first."""
        ce = CostEstimator()
        estimates = ce.estimate_for_gpu("A100-80GB", training_hours=10.0)
        for i in range(len(estimates) - 1):
            assert estimates[i].total_cost <= estimates[i + 1].total_cost

    def test_spot_pricing_cheaper(self):
        """Spot pricing should be cheaper than on-demand where available."""
        ce = CostEstimator()
        estimates = ce.estimate_for_gpu("A100-80GB", training_hours=10.0)
        for e in estimates:
            if e.spot_total_cost is not None:
                assert e.spot_total_cost < e.total_cost

    def test_longer_training_costs_more(self):
        """More training hours should cost more."""
        ce = CostEstimator()
        cost_5h = ce.estimate_for_gpu("A100-80GB", training_hours=5.0)
        cost_20h = ce.estimate_for_gpu("A100-80GB", training_hours=20.0)
        # Compare same provider
        if cost_5h and cost_20h:
            assert cost_20h[0].total_cost > cost_5h[0].total_cost

    def test_h100_more_expensive_than_t4(self):
        """H100 should cost more per hour than T4."""
        ce = CostEstimator()
        h100 = ce.estimate_for_gpu("H100-80GB", training_hours=1.0)
        t4 = ce.estimate_for_gpu("T4-16GB", training_hours=1.0)
        if h100 and t4:
            assert min(e.hourly_rate for e in h100) > min(e.hourly_rate for e in t4)

    def test_unknown_gpu_returns_empty(self):
        """Unknown GPU should return empty list, not error."""
        ce = CostEstimator()
        estimates = ce.estimate_for_gpu("FAKE-GPU-999GB", training_hours=5.0)
        assert estimates == []

    def test_quick_estimate(self):
        """Quick estimate should return a CostComparison."""
        ce = CostEstimator()
        comparison = ce.quick_estimate("A100-80GB", training_hours=5.0)
        assert comparison.training_hours == 5.0
        assert len(comparison.estimates) > 0
        assert comparison.cheapest is not None

    def test_compare_all(self):
        """Full comparison across GPUs should aggregate results."""
        ce = CostEstimator()
        comparison = ce.compare_all(
            compatible_gpus=["A100-80GB", "H100-80GB"],
            training_hours_per_gpu={"A100-80GB": 10.0, "H100-80GB": 5.0},
        )
        assert len(comparison.estimates) > 0
        assert comparison.cheapest is not None


class TestEstimatorCostIntegration:
    """Test cost estimation through the main Estimator API."""

    def test_estimate_costs_auto_time(self):
        """estimate_costs should auto-calculate training time."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        costs = est.estimate_costs(
            gpu="A100-80GB", dataset_size=50000, epochs=3,
        )
        assert len(costs.estimates) > 0
        assert costs.training_hours > 0

    def test_estimate_costs_manual_hours(self):
        """estimate_costs should accept pre-computed training hours."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        costs = est.estimate_costs(gpu="A100-80GB", training_hours=5.0)
        assert costs.training_hours == 5.0
        for e in costs.estimates:
            assert e.training_hours == 5.0

    def test_full_comparison(self):
        """full_comparison should check all compatible GPUs."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        comparison = est.full_comparison(dataset_size=50000, epochs=3)
        assert len(comparison.estimates) > 0
        assert comparison.cheapest is not None
        # Should have estimates from multiple providers
        providers = set(e.provider for e in comparison.estimates)
        assert len(providers) > 1

    def test_full_comparison_sorted_by_cost(self):
        """Full comparison results should be sorted by cost."""
        est = Estimator(
            model="meta-llama/Llama-3.1-8B", method="qlora",
            quantization="4bit", batch_size=4, seq_length=2048,
        )
        comparison = est.full_comparison(dataset_size=50000, epochs=3)
        costs = [e.total_cost for e in comparison.estimates]
        assert costs == sorted(costs)
