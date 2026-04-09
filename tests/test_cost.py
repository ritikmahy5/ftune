"""Tests for cloud cost estimation engine."""

import shutil
from pathlib import Path

import pytest

from ftune import Estimator
from ftune.core.cost import (
    CostEstimator, list_providers, get_provider_display_name,
    update_price, get_staleness,
)


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


# ─────────────────────────────────────────────────────────
# Pricing update tests
# ─────────────────────────────────────────────────────────

_PRICING_SRC = Path(__file__).parent.parent / "src" / "ftune" / "data" / "pricing.yaml"


@pytest.fixture
def tmp_pricing(tmp_path: Path) -> Path:
    """Copy real pricing.yaml to a temp dir for safe mutation."""
    dst = tmp_path / "pricing.yaml"
    shutil.copy(_PRICING_SRC, dst)
    return dst


class TestUpdatePrice:
    """Tests for the pricing update mechanism."""

    def test_update_valid_price(self, tmp_pricing: Path):
        """Should update rate and return old/new values."""
        old, new = update_price("lambda_labs", "A100-80GB", 2.99, pricing_path=tmp_pricing)
        assert old == 1.99
        assert new == 2.99

        # Verify written to disk
        import yaml
        with open(tmp_pricing) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["lambda_labs"]["gpus"]["A100-80GB"]["hourly_rate"] == 2.99

    def test_update_sets_provider_timestamp(self, tmp_pricing: Path):
        """Should update provider's last_updated to today."""
        from datetime import date
        update_price("lambda_labs", "A100-80GB", 2.99, pricing_path=tmp_pricing)

        import yaml
        with open(tmp_pricing) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["lambda_labs"]["last_updated"] == date.today().isoformat()

    def test_update_sets_global_timestamp(self, tmp_pricing: Path):
        """Global last_updated should be max of all provider dates."""
        from datetime import date
        update_price("lambda_labs", "A100-80GB", 2.99, pricing_path=tmp_pricing)

        import yaml
        with open(tmp_pricing) as f:
            data = yaml.safe_load(f)
        assert data["last_updated"] == date.today().isoformat()

    def test_update_with_spot_rate(self, tmp_pricing: Path):
        """Should update both on-demand and spot rates."""
        update_price("runpod", "A100-80GB", 1.89, spot_hourly_rate=1.19, pricing_path=tmp_pricing)

        import yaml
        with open(tmp_pricing) as f:
            data = yaml.safe_load(f)
        gpu = data["providers"]["runpod"]["gpus"]["A100-80GB"]
        assert gpu["hourly_rate"] == 1.89
        assert gpu["spot_hourly_rate"] == 1.19

    def test_update_invalid_provider(self, tmp_pricing: Path):
        """Unknown provider should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            update_price("fake_provider", "A100-80GB", 1.99, pricing_path=tmp_pricing)

    def test_update_invalid_gpu(self, tmp_pricing: Path):
        """Unknown GPU for valid provider should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            update_price("lambda_labs", "FAKE-GPU-999GB", 1.99, pricing_path=tmp_pricing)

    def test_update_negative_rate(self, tmp_pricing: Path):
        """Negative rate should raise ValueError."""
        with pytest.raises(ValueError, match="between"):
            update_price("lambda_labs", "A100-80GB", -1.0, pricing_path=tmp_pricing)

    def test_update_zero_rate(self, tmp_pricing: Path):
        """Zero rate should raise ValueError."""
        with pytest.raises(ValueError, match="between"):
            update_price("lambda_labs", "A100-80GB", 0.0, pricing_path=tmp_pricing)

    def test_update_extreme_rate(self, tmp_pricing: Path):
        """Rate > $500 should raise ValueError."""
        with pytest.raises(ValueError, match="between"):
            update_price("lambda_labs", "A100-80GB", 999.0, pricing_path=tmp_pricing)

    def test_spot_rate_above_ondemand(self, tmp_pricing: Path):
        """Spot rate >= on-demand should raise ValueError."""
        with pytest.raises(ValueError, match="less than"):
            update_price("runpod", "A100-80GB", 1.50, spot_hourly_rate=2.00, pricing_path=tmp_pricing)


class TestStaleness:
    """Tests for pricing staleness reporting."""

    def test_get_staleness_returns_all_providers(self):
        """Should return info for all providers."""
        staleness = get_staleness()
        providers = list_providers()
        assert len(staleness) == len(providers)

    def test_staleness_has_required_fields(self):
        """Each entry should have provider, display_name, last_updated, days_ago, gpu_count."""
        staleness = get_staleness()
        for s in staleness:
            assert "provider" in s
            assert "display_name" in s
            assert "last_updated" in s
            assert "days_ago" in s
            assert "gpu_count" in s
            assert s["days_ago"] >= 0
            assert s["gpu_count"] > 0

    def test_staleness_sorted_most_stale_first(self):
        """Should be sorted by days_ago descending."""
        staleness = get_staleness()
        days = [s["days_ago"] for s in staleness]
        assert days == sorted(days, reverse=True)

    def test_staleness_after_update(self, tmp_pricing: Path):
        """Updated provider should show 0 days ago."""
        update_price("lambda_labs", "A100-80GB", 2.99, pricing_path=tmp_pricing)
        staleness = get_staleness(pricing_path=tmp_pricing)
        lambda_entry = next(s for s in staleness if s["provider"] == "lambda_labs")
        assert lambda_entry["days_ago"] == 0
