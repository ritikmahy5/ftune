"""Cloud cost estimation engine for LLM fine-tuning.

Compares training costs across cloud GPU providers using a
bundled pricing database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ftune.core.models import CostComparison, CostEstimate

_DATA_DIR = Path(__file__).parent.parent / "data"
_pricing_cache: Optional[dict] = None


def _load_pricing() -> dict:
    """Load pricing database from YAML."""
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache

    path = _DATA_DIR / "pricing.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Pricing file not found: {path}")

    with open(path, "r") as f:
        _pricing_cache = yaml.safe_load(f)
    return _pricing_cache


def list_providers() -> List[str]:
    """Return list of available provider keys."""
    data = _load_pricing()
    return sorted(data.get("providers", {}).keys())


def get_provider_display_name(provider_key: str) -> str:
    """Get the human-readable name for a provider."""
    data = _load_pricing()
    provider = data.get("providers", {}).get(provider_key, {})
    return provider.get("display_name", provider_key)


def get_provider_gpus(provider_key: str) -> Dict[str, dict]:
    """Get available GPUs and pricing for a provider."""
    data = _load_pricing()
    provider = data.get("providers", {}).get(provider_key, {})
    return provider.get("gpus", {})


class CostEstimator:
    """Estimates cloud costs for LLM fine-tuning across providers.

    Given a training time estimate and GPU requirement, compares costs
    across all supported cloud providers.
    """

    def __init__(self) -> None:
        self.pricing = _load_pricing()

    def estimate_for_gpu(
        self,
        gpu_name: str,
        training_hours: float,
        num_gpus: int = 1,
        include_spot: bool = True,
    ) -> List[CostEstimate]:
        """Get cost estimates for a specific GPU across all providers.

        Args:
            gpu_name: GPU name (e.g. 'A100-80GB').
            training_hours: Estimated training duration in hours.
            num_gpus: Number of GPUs needed.
            include_spot: Whether to include spot/preemptible pricing.

        Returns:
            List of CostEstimate sorted by total cost (cheapest first).
        """
        estimates = []
        providers = self.pricing.get("providers", {})

        for provider_key, provider_data in providers.items():
            gpus = provider_data.get("gpus", {})
            if gpu_name not in gpus:
                continue

            gpu_pricing = gpus[gpu_name]
            hourly = gpu_pricing["hourly_rate"]
            gpu_count = gpu_pricing.get("gpu_count", 1)

            # Scale by number of GPUs needed vs what the instance provides
            instances_needed = max(1, num_gpus // gpu_count)
            total_hourly = hourly * instances_needed
            total_cost = total_hourly * training_hours

            spot_hourly = gpu_pricing.get("spot_hourly_rate")
            spot_total = None
            if spot_hourly and include_spot:
                spot_total = spot_hourly * instances_needed * training_hours

            estimates.append(CostEstimate(
                provider=provider_data.get("display_name", provider_key),
                gpu=gpu_name,
                gpu_count=gpu_count * instances_needed,
                hourly_rate=round(total_hourly, 2),
                training_hours=round(training_hours, 2),
                total_cost=round(total_cost, 2),
                spot_hourly_rate=round(spot_hourly, 2) if spot_hourly else None,
                spot_total_cost=round(spot_total, 2) if spot_total else None,
                instance_type=gpu_pricing.get("instance_type", ""),
                region=gpu_pricing.get("region", ""),
            ))

        # Sort by total cost
        estimates.sort(key=lambda e: e.total_cost)
        return estimates

    def compare_all(
        self,
        compatible_gpus: List[str],
        training_hours_per_gpu: Dict[str, float],
        num_gpus: int = 1,
        include_spot: bool = True,
    ) -> CostComparison:
        """Compare costs across all compatible GPUs and providers.

        Args:
            compatible_gpus: List of GPU names that can handle the workload.
            training_hours_per_gpu: Dict mapping GPU name → estimated hours.
            num_gpus: Number of GPUs needed.
            include_spot: Whether to include spot pricing.

        Returns:
            CostComparison with all estimates and recommendations.
        """
        all_estimates = []

        for gpu_name in compatible_gpus:
            hours = training_hours_per_gpu.get(gpu_name, 0)
            if hours <= 0:
                continue

            gpu_estimates = self.estimate_for_gpu(
                gpu_name, hours, num_gpus, include_spot
            )
            all_estimates.extend(gpu_estimates)

        # Sort all by total on-demand cost
        all_estimates.sort(key=lambda e: e.total_cost)

        # Find cheapest and best value
        cheapest = all_estimates[0].provider if all_estimates else None

        # Best value considers spot pricing too
        best_value = cheapest
        if all_estimates:
            spot_estimates = [e for e in all_estimates if e.spot_total_cost is not None]
            if spot_estimates:
                best_spot = min(spot_estimates, key=lambda e: e.spot_total_cost)
                if best_spot.spot_total_cost < all_estimates[0].total_cost:
                    best_value = f"{best_spot.provider} (spot)"

        # Use the min training hours for the comparison summary
        min_hours = min(training_hours_per_gpu.values()) if training_hours_per_gpu else 0

        return CostComparison(
            estimates=all_estimates,
            training_hours=round(min_hours, 2),
            cheapest=cheapest,
            best_value=best_value,
        )

    def quick_estimate(
        self, gpu_name: str, training_hours: float
    ) -> CostComparison:
        """Quick cost comparison for a single GPU type.

        Convenience method when you already know the GPU and hours.

        Args:
            gpu_name: GPU name (e.g. 'A100-80GB').
            training_hours: Estimated training hours.

        Returns:
            CostComparison for this GPU across all providers.
        """
        estimates = self.estimate_for_gpu(gpu_name, training_hours)

        cheapest = estimates[0].provider if estimates else None
        best_value = cheapest
        if estimates:
            spot_options = [e for e in estimates if e.spot_total_cost is not None]
            if spot_options:
                best_spot = min(spot_options, key=lambda e: e.spot_total_cost)
                if best_spot.spot_total_cost < estimates[0].total_cost:
                    best_value = f"{best_spot.provider} (spot)"

        return CostComparison(
            estimates=estimates,
            training_hours=round(training_hours, 2),
            cheapest=cheapest,
            best_value=best_value,
        )
