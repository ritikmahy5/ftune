"""Cloud cost estimation engine for LLM fine-tuning.

Compares training costs across cloud GPU providers using a
bundled pricing database. Supports updating prices via CLI.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from ftune.core.models import CostComparison, CostEstimate

_DATA_DIR = Path(__file__).parent.parent / "data"
_PRICING_PATH = _DATA_DIR / "pricing.yaml"
_pricing_cache: Optional[dict] = None

# Header comment re-added after YAML dump (pyyaml strips comments)
_PRICING_HEADER = """\
# ftune Cloud GPU Pricing Database
# Prices are per-GPU per-hour in USD.
# Sources: Official provider pricing pages.
# Use `ftune pricing-update <provider> <gpu> --rate <price>` to update.
"""


def _load_pricing(path: Path | None = None) -> dict:
    """Load pricing database from YAML."""
    global _pricing_cache
    target = path or _PRICING_PATH
    if path is None and _pricing_cache is not None:
        return _pricing_cache

    if not target.exists():
        raise FileNotFoundError(f"Pricing file not found: {target}")

    with open(target, "r") as f:
        data = yaml.safe_load(f)

    if path is None:
        _pricing_cache = data
    return data


def _save_pricing(data: dict, path: Path | None = None) -> None:
    """Write pricing database back to YAML."""
    global _pricing_cache
    target = path or _PRICING_PATH

    yaml_content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    with open(target, "w") as f:
        f.write(_PRICING_HEADER)
        f.write("\n")
        f.write(yaml_content)

    if path is None:
        _pricing_cache = data


def update_price(
    provider: str,
    gpu: str,
    hourly_rate: float,
    spot_hourly_rate: float | None = None,
    pricing_path: Path | None = None,
) -> tuple[float, float]:
    """Update a GPU price for a provider.

    Args:
        provider: Provider key (e.g. 'lambda_labs', 'runpod').
        gpu: GPU name (e.g. 'A100-80GB').
        hourly_rate: New hourly rate in USD.
        spot_hourly_rate: New spot hourly rate in USD (optional).
        pricing_path: Override path for testing.

    Returns:
        Tuple of (old_rate, new_rate).

    Raises:
        KeyError: If provider or GPU not found.
        ValueError: If rate is invalid.
    """
    if hourly_rate <= 0 or hourly_rate > 500:
        raise ValueError(
            f"Hourly rate must be between $0 and $500, got ${hourly_rate:.2f}"
        )
    if spot_hourly_rate is not None:
        if spot_hourly_rate <= 0:
            raise ValueError(f"Spot rate must be > $0, got ${spot_hourly_rate:.2f}")
        if spot_hourly_rate >= hourly_rate:
            raise ValueError(
                f"Spot rate (${spot_hourly_rate:.2f}) must be less than "
                f"on-demand rate (${hourly_rate:.2f})"
            )

    data = _load_pricing(pricing_path)
    providers = data.get("providers", {})

    if provider not in providers:
        available = ", ".join(sorted(providers.keys()))
        raise KeyError(f"Provider '{provider}' not found. Available: {available}")

    gpus = providers[provider].get("gpus", {})
    if gpu not in gpus:
        available = ", ".join(sorted(gpus.keys()))
        raise KeyError(
            f"GPU '{gpu}' not found for {provider}. Available: {available}"
        )

    old_rate = gpus[gpu]["hourly_rate"]
    gpus[gpu]["hourly_rate"] = hourly_rate

    if spot_hourly_rate is not None:
        gpus[gpu]["spot_hourly_rate"] = spot_hourly_rate

    # Update provider timestamp
    today = date.today().isoformat()
    providers[provider]["last_updated"] = today

    # Update global timestamp to max of all providers
    all_dates = [
        p.get("last_updated", "2025-01-01") for p in providers.values()
    ]
    data["last_updated"] = max(all_dates)

    _save_pricing(data, pricing_path)
    return old_rate, hourly_rate


def get_staleness(pricing_path: Path | None = None) -> list[dict]:
    """Return staleness info for all providers.

    Returns:
        List of dicts with keys: provider, display_name, last_updated,
        days_ago, gpu_count. Sorted by most stale first.
    """
    data = _load_pricing(pricing_path)
    providers = data.get("providers", {})
    today = date.today()
    result = []

    for key, pdata in providers.items():
        last_str = pdata.get("last_updated", data.get("last_updated", "2025-01-01"))
        last_date = datetime.strptime(last_str, "%Y-%m-%d").date()
        days_ago = (today - last_date).days
        gpu_count = len(pdata.get("gpus", {}))

        result.append({
            "provider": key,
            "display_name": pdata.get("display_name", key),
            "last_updated": last_str,
            "days_ago": days_ago,
            "gpu_count": gpu_count,
        })

    result.sort(key=lambda x: -x["days_ago"])
    return result


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

    @staticmethod
    def _find_best_value(
        estimates: List[CostEstimate],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Find cheapest and best-value providers from a list of estimates.

        Returns:
            Tuple of (cheapest_provider, best_value_provider).
        """
        if not estimates:
            return None, None
        cheapest = estimates[0].provider
        best_value = cheapest
        spot_estimates = [e for e in estimates if e.spot_total_cost is not None]
        if spot_estimates:
            best_spot = min(spot_estimates, key=lambda e: e.spot_total_cost or float('inf'))
            if best_spot.spot_total_cost is not None and best_spot.spot_total_cost < estimates[0].total_cost:
                best_value = f"{best_spot.provider} (spot)"
        return cheapest, best_value

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
        cheapest, best_value = self._find_best_value(all_estimates)

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

        cheapest, best_value = self._find_best_value(estimates)

        return CostComparison(
            estimates=estimates,
            training_hours=round(training_hours, 2),
            cheapest=cheapest,
            best_value=best_value,
        )
