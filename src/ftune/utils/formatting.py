"""Formatting and display helpers."""

from __future__ import annotations


def format_bytes_gb(bytes_val: float) -> str:
    """Format byte count as human-readable GB string."""
    gb = bytes_val / (1024**3)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    mb = bytes_val / (1024**2)
    return f"{mb:.1f} MB"


def format_params(count: int) -> str:
    """Format parameter count with human-readable suffix.

    Examples:
        7_000_000_000 → '7.00B'
        3_800_000 → '3.80M'
        150_000 → '150.00K'
    """
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def format_percentage(value: float) -> str:
    """Format a percentage value."""
    if value < 0.01:
        return f"{value:.4f}%"
    elif value < 1.0:
        return f"{value:.3f}%"
    return f"{value:.1f}%"
