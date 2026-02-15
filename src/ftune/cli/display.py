"""Rich display helpers for beautiful terminal output."""

from __future__ import annotations

from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ftune.core.models import CostComparison, GPUFit, MemoryBreakdown, TimeEstimate
from ftune.utils.formatting import format_params, format_percentage

console = Console()


def print_header():
    """Print the ftune header banner."""
    console.print(
        Panel(
            "[bold cyan]⚡ ftune[/bold cyan] — LLM Fine-Tuning Cost Estimator",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


def print_memory(mem: MemoryBreakdown, model_name: str, config_summary: str):
    """Print memory estimation results with Rich formatting."""
    console.print(f"[bold]📦 Model:[/bold] {model_name}")
    console.print(f"[bold]⚙️  Config:[/bold] {config_summary}")
    console.print()

    table = Table(
        title="Memory Breakdown",
        title_style="bold",
        border_style="dim",
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("Component", style="white", min_width=22)
    table.add_column("VRAM (GB)", justify="right", style="cyan", min_width=12)
    table.add_column("% of Total", justify="right", style="dim", min_width=10)

    components = [
        ("Base Model Weights", mem.model_weights_gb),
        ("LoRA Adapters", mem.trainable_params_gb),
        ("Gradients", mem.gradients_gb),
        ("Optimizer States", mem.optimizer_states_gb),
        ("Activations", mem.activations_gb),
        ("CUDA Overhead", mem.overhead_gb),
    ]

    for name, gb in components:
        pct = (gb / mem.total_gb * 100) if mem.total_gb > 0 else 0
        table.add_row(name, f"{gb:.3f}", f"{pct:.1f}%")

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold cyan]{mem.total_gb:.2f}[/bold cyan]",
        "[bold]100%[/bold]",
    )

    console.print(table)
    console.print()

    # Trainable params
    console.print(
        f"  [dim]Trainable:[/dim] {format_params(mem.trainable_params)} "
        f"({format_percentage(mem.trainable_percentage)}) of "
        f"{format_params(mem.total_params)} total"
    )
    console.print()


def print_gpu_fit(fits: List[GPUFit]):
    """Print GPU compatibility results."""
    compatible = [g for g in fits if g.fits]
    incompatible = [g for g in fits if not g.fits]

    if compatible:
        names = ", ".join(
            f"{g.gpu_name} ({g.utilization_percent:.0f}%)" for g in compatible
        )
        console.print(f"  [green]✅ Fits on:[/green] {names}")

    if incompatible:
        names = ", ".join(g.gpu_name for g in incompatible)
        console.print(f"  [red]❌ Too large for:[/red] {names}")

    console.print()


def print_time(times: List[TimeEstimate], dataset_size: int, epochs: int):
    """Print training time estimates."""
    table = Table(
        title=f"Training Time — {dataset_size:,} samples × {epochs} epochs",
        title_style="bold",
        border_style="dim",
    )
    table.add_column("GPU", style="white", min_width=18)
    table.add_column("Time/Epoch", justify="right", style="yellow", min_width=12)
    table.add_column("Total", justify="right", style="bold cyan", min_width=12)
    table.add_column("Steps", justify="right", style="dim", min_width=10)

    for t in times:
        table.add_row(
            t.gpu_name,
            f"{t.hours_per_epoch:.1f}h",
            f"{t.total_hours:.1f}h",
            f"{t.total_steps:,}",
        )

    console.print(table)
    console.print()


def print_costs(comparison: CostComparison, limit: int = 15):
    """Print cost comparison table."""
    table = Table(
        title="Cost Comparison — All Providers",
        title_style="bold",
        border_style="dim",
    )
    table.add_column("Provider", style="white", min_width=16)
    table.add_column("GPU", style="white", min_width=16)
    table.add_column("Hours", justify="right", style="dim", min_width=8)
    table.add_column("$/hr", justify="right", style="yellow", min_width=8)
    table.add_column("Total", justify="right", style="bold green", min_width=10)
    table.add_column("Spot Total", justify="right", style="cyan", min_width=10)

    for e in comparison.estimates[:limit]:
        spot = f"${e.spot_total_cost:.2f}" if e.spot_total_cost else "—"
        table.add_row(
            e.provider,
            e.gpu,
            f"{e.training_hours:.1f}h",
            f"${e.hourly_rate:.2f}",
            f"${e.total_cost:.2f}",
            spot,
        )

    console.print(table)
    console.print()

    if comparison.cheapest:
        console.print(f"  [green]💡 Cheapest on-demand:[/green] [bold]{comparison.cheapest}[/bold]")
    if comparison.best_value:
        console.print(f"  [cyan]🏆 Best value:[/cyan] [bold]{comparison.best_value}[/bold]")
    console.print()
