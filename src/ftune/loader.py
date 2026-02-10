"""Load model and GPU specs from bundled YAML databases."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ftune.core.models import GPUSpec, ModelSpec

_DATA_DIR = Path(__file__).parent / "data"

# Module-level caches
_models_cache: Optional[Dict[str, ModelSpec]] = None
_gpus_cache: Optional[Dict[str, GPUSpec]] = None


def _load_yaml(filename: str) -> dict:
    """Load a YAML file from the data directory."""
    path = _DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_models() -> Dict[str, ModelSpec]:
    """Load all model specs from models.yaml.

    Returns:
        Dictionary mapping model name → ModelSpec.
    """
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    data = _load_yaml("models.yaml")
    models = {}
    for name, spec in data.get("models", {}).items():
        models[name] = ModelSpec(name=name, **spec)

    _models_cache = models
    return models


def load_gpus() -> Dict[str, GPUSpec]:
    """Load all GPU specs from gpus.yaml.

    Returns:
        Dictionary mapping GPU name → GPUSpec.
    """
    global _gpus_cache
    if _gpus_cache is not None:
        return _gpus_cache

    data = _load_yaml("gpus.yaml")
    gpus = {}
    for name, spec in data.get("gpus", {}).items():
        gpus[name] = GPUSpec(name=name, **spec)

    _gpus_cache = gpus
    return gpus


def get_model(name: str) -> ModelSpec:
    """Get a specific model spec by name.

    Supports fuzzy matching: tries exact match first, then case-insensitive,
    then partial match on the model name suffix.

    Args:
        name: Model name (e.g. 'meta-llama/Llama-3.1-8B' or 'Llama-3.1-8B').

    Returns:
        ModelSpec for the matched model.

    Raises:
        KeyError: If no matching model is found.
    """
    models = load_models()

    # Exact match
    if name in models:
        return models[name]

    # Case-insensitive match
    name_lower = name.lower()
    for key, spec in models.items():
        if key.lower() == name_lower:
            return spec

    # Partial match on suffix (e.g., "Llama-3.1-8B" matches "meta-llama/Llama-3.1-8B")
    for key, spec in models.items():
        if key.lower().endswith(name_lower) or name_lower in key.lower():
            return spec

    # No match in local DB — try HuggingFace Hub
    try:
        from ftune.hub import resolve_model_from_hub
        spec = resolve_model_from_hub(name)
        # Cache it for subsequent calls
        models[name] = spec
        return spec
    except (ValueError, ConnectionError, Exception):
        pass

    # Nothing worked
    available = ", ".join(sorted(models.keys()))
    raise KeyError(
        f"Model '{name}' not found in local database or HuggingFace Hub. "
        f"Available local models: {available}. "
        f"For HuggingFace models, use the full model ID (e.g. 'org/model-name') "
        f"and ensure you have internet access."
    )


def get_gpu(name: str) -> GPUSpec:
    """Get a specific GPU spec by name with fuzzy matching.

    Args:
        name: GPU name (e.g. 'A100-80GB' or 'a100').

    Returns:
        GPUSpec for the matched GPU.

    Raises:
        KeyError: If no matching GPU is found.
    """
    gpus = load_gpus()

    # Exact match
    if name in gpus:
        return gpus[name]

    # Case-insensitive match
    name_lower = name.lower().replace(" ", "").replace("_", "-")
    for key, spec in gpus.items():
        key_normalized = key.lower().replace(" ", "").replace("_", "-")
        if key_normalized == name_lower or name_lower in key_normalized:
            return spec

    available = ", ".join(sorted(gpus.keys()))
    raise KeyError(
        f"GPU '{name}' not found in database. "
        f"Available GPUs: {available}"
    )


def list_model_names() -> List[str]:
    """Return sorted list of all available model names."""
    return sorted(load_models().keys())


def list_gpu_names() -> List[str]:
    """Return sorted list of all available GPU names."""
    return sorted(load_gpus().keys())
