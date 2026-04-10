"""Tests for the Budget Optimizer module."""

from ftuneai.optimizer import BudgetOptimizer


class TestBudgetConstraints:
    """Tests that budget constraints are respected."""

    def test_results_within_budget(self):
        """All recommendations should respect the budget limit."""
        recs = BudgetOptimizer.optimize(
            model="meta-llama/Llama-3.1-8B",
            budget=50.0,
            dataset_size=10000,
            epochs=1,
            priority="cost",
        )
        for r in recs:
            assert r.estimated_cost <= 50.0, (
                f"Recommendation cost ${r.estimated_cost} exceeds budget $50"
            )

    def test_tiny_budget_returns_few_or_no_results(self):
        """A very small budget should return few or no recommendations."""
        recs = BudgetOptimizer.optimize(
            model="meta-llama/Llama-3.1-8B",
            budget=0.01,
            dataset_size=50000,
            epochs=3,
        )
        # With $0.01 budget, most configs won't fit
        assert len(recs) <= 3

    def test_no_budget_constraint(self):
        """With no budget limit, should return multiple recommendations."""
        recs = BudgetOptimizer.optimize(
            model="meta-llama/Llama-3.1-8B",
            dataset_size=10000,
            epochs=1,
            budget=None,
        )
        assert len(recs) > 0


class TestGPUConstraints:
    """Tests for GPU-specific constraints."""

    def test_specific_gpu(self):
        """Results should only reference the specified GPU."""
        recs = BudgetOptimizer.optimize(
            model="meta-llama/Llama-3.1-8B",
            gpu="A100-80GB",
            dataset_size=10000,
            epochs=1,
        )
        for r in recs:
            assert r.gpu == "A100-80GB"

    def test_memory_fits_gpu(self):
        """Estimated memory should not exceed GPU VRAM."""
        recs = BudgetOptimizer.optimize(
            model="meta-llama/Llama-3.1-8B",
            gpu="RTX-4090-24GB",
            dataset_size=10000,
            epochs=1,
        )
        for r in recs:
            assert r.estimated_memory_gb <= 24.0


class TestPrioritySorting:
    """Tests that results are sorted by the requested priority."""

    def test_cost_priority(self):
        """Cost priority should sort by estimated cost ascending."""
        recs = BudgetOptimizer.optimize(
            model="meta-llama/Llama-3.1-8B",
            gpu="A100-80GB",
            dataset_size=10000,
            epochs=1,
            priority="cost",
        )
        if len(recs) >= 2:
            for i in range(len(recs) - 1):
                assert recs[i].estimated_cost <= recs[i + 1].estimated_cost

    def test_speed_priority(self):
        """Speed priority should sort by estimated hours ascending."""
        recs = BudgetOptimizer.optimize(
            model="meta-llama/Llama-3.1-8B",
            gpu="A100-80GB",
            dataset_size=10000,
            epochs=1,
            priority="speed",
        )
        if len(recs) >= 2:
            for i in range(len(recs) - 1):
                assert recs[i].estimated_hours <= recs[i + 1].estimated_hours


class TestEdgeCases:
    """Tests for edge cases."""

    def test_max_results_capped_at_10(self):
        """Should return at most 10 recommendations."""
        recs = BudgetOptimizer.optimize(
            model="meta-llama/Llama-3.1-8B",
            dataset_size=10000,
            epochs=1,
        )
        assert len(recs) <= 10

    def test_all_recs_have_required_fields(self):
        """All recommendations should have populated fields."""
        recs = BudgetOptimizer.optimize(
            model="meta-llama/Llama-3.1-8B",
            gpu="A100-80GB",
            dataset_size=10000,
            epochs=1,
        )
        for r in recs:
            assert r.method
            assert r.gpu
            assert r.estimated_hours > 0
            assert r.estimated_memory_gb > 0
            assert r.fits is True

    def test_format_recommendations_empty(self):
        """Formatting empty list should return a message."""
        output = BudgetOptimizer.format_recommendations([])
        assert "No configurations found" in output

    def test_format_recommendations_non_empty(self):
        """Formatting results should produce readable output."""
        recs = BudgetOptimizer.optimize(
            model="meta-llama/Llama-3.1-8B",
            gpu="A100-80GB",
            dataset_size=10000,
            epochs=1,
        )
        if recs:
            output = BudgetOptimizer.format_recommendations(recs)
            assert "ftune" in output
            assert "$" in output
