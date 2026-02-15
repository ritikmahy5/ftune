"""ftune CLI — Estimate GPU memory, training time, and costs from your terminal.

Install: pip install ftune[cli]
Usage:   ftune estimate --model meta-llama/Llama-3.1-8B --method qlora
"""

from __future__ import annotations

from typing import Optional

import typer

from ftune import Estimator, list_model_names, list_gpu_names

app = typer.Typer(
    name="ftune",
    help="⚡ Estimate GPU memory, training time, and costs for LLM fine-tuning.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def estimate(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g. 'meta-llama/Llama-3.1-8B')"),
    method: str = typer.Option("qlora", "--method", help="Fine-tuning method: full, lora, qlora"),
    quantization: str = typer.Option("4bit", "--quantization", "-q", help="Quantization: none, 4bit, 8bit"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Per-device batch size"),
    seq_length: int = typer.Option(2048, "--seq-length", "-s", help="Max sequence length"),
    lora_rank: int = typer.Option(16, "--lora-rank", "-r", help="LoRA rank"),
    lora_alpha: int = typer.Option(32, "--lora-alpha", help="LoRA alpha"),
    lora_target: str = typer.Option("attention", "--lora-target", help="LoRA targets: attention, attention_all, all_linear"),
    optimizer: str = typer.Option("adamw", "--optimizer", help="Optimizer: adamw, adam, sgd, adam_8bit, adafactor"),
    gradient_checkpointing: bool = typer.Option(True, "--grad-ckpt/--no-grad-ckpt", help="Gradient checkpointing"),
    dataset_size: int = typer.Option(50000, "--dataset-size", "-d", help="Number of training samples"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs"),
    num_gpus: int = typer.Option(1, "--num-gpus", "-g", help="Number of GPUs"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json, markdown"),
):
    """Estimate memory, training time, and costs for fine-tuning."""
    from ftune.cli.display import (
        console, print_header, print_memory, print_gpu_fit, print_time, print_costs,
    )

    # Handle qlora default quantization
    if method == "qlora" and quantization == "none":
        quantization = "4bit"
    if method == "full":
        quantization = "none"

    try:
        est = Estimator(
            model=model,
            method=method,
            quantization=quantization,
            batch_size=batch_size,
            seq_length=seq_length,
            gradient_checkpointing=gradient_checkpointing,
            optimizer=optimizer,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target=lora_target,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # JSON output
    if output == "json":
        import json
        mem = est.estimate_memory()
        times = est.estimate_time_all_gpus(dataset_size=dataset_size, epochs=epochs, num_gpus=num_gpus)
        costs = est.full_comparison(dataset_size=dataset_size, epochs=epochs, num_gpus=num_gpus)

        result = {
            "model": model,
            "method": method,
            "quantization": quantization,
            "memory": {
                "total_gb": mem.total_gb,
                "model_weights_gb": mem.model_weights_gb,
                "trainable_params_gb": mem.trainable_params_gb,
                "gradients_gb": mem.gradients_gb,
                "optimizer_states_gb": mem.optimizer_states_gb,
                "activations_gb": mem.activations_gb,
                "overhead_gb": mem.overhead_gb,
                "trainable_params": mem.trainable_params,
                "trainable_percentage": mem.trainable_percentage,
            },
            "training_time": [
                {"gpu": t.gpu_name, "total_hours": t.total_hours, "hours_per_epoch": t.hours_per_epoch}
                for t in times
            ],
            "costs": [
                {
                    "provider": c.provider,
                    "gpu": c.gpu,
                    "total_cost": c.total_cost,
                    "hourly_rate": c.hourly_rate,
                    "spot_total_cost": c.spot_total_cost,
                }
                for c in costs.estimates
            ],
        }
        console.print_json(json.dumps(result, indent=2))
        return

    # Rich output
    print_header()

    # Config summary
    config_str = f"{method.upper()}"
    if method != "full":
        config_str += f" (rank={lora_rank}, alpha={lora_alpha})"
    if quantization != "none":
        config_str += f", {quantization}"
    config_str += f" | batch={batch_size}, seq={seq_length}"
    if gradient_checkpointing:
        config_str += " | grad-ckpt"

    # Memory
    mem = est.estimate_memory()
    print_memory(mem, model, config_str)

    # GPU fit
    fits = est.check_gpu_fit()
    print_gpu_fit(fits)

    # Training time
    times = est.estimate_time_all_gpus(
        dataset_size=dataset_size, epochs=epochs, num_gpus=num_gpus,
    )
    if times:
        print_time(times, dataset_size, epochs)

    # Cost comparison
    costs = est.full_comparison(
        dataset_size=dataset_size, epochs=epochs, num_gpus=num_gpus,
    )
    if costs.estimates:
        print_costs(costs)


@app.command()
def models():
    """List all supported models."""
    from rich.table import Table
    from ftune.cli.display import console
    from ftune.loader import load_models
    from ftune.utils.formatting import format_params

    all_models = load_models()

    table = Table(title="Supported Models", border_style="dim")
    table.add_column("Model", style="cyan")
    table.add_column("Parameters", justify="right")
    table.add_column("Hidden", justify="right", style="dim")
    table.add_column("Layers", justify="right", style="dim")
    table.add_column("Max Seq", justify="right", style="dim")

    for name, spec in sorted(all_models.items()):
        table.add_row(
            name,
            format_params(spec.parameters),
            str(spec.hidden_size),
            str(spec.num_layers),
            f"{spec.max_seq_length:,}",
        )

    console.print(table)
    console.print(f"\n  [dim]Plus any model on HuggingFace Hub via auto-detect[/dim]")


@app.command()
def gpus():
    """List all supported GPUs."""
    from rich.table import Table
    from ftune.cli.display import console
    from ftune.loader import load_gpus

    all_gpus = load_gpus()

    table = Table(title="Supported GPUs", border_style="dim")
    table.add_column("GPU", style="cyan")
    table.add_column("VRAM", justify="right")
    table.add_column("FP16 TFLOPS", justify="right", style="yellow")
    table.add_column("BF16 TFLOPS", justify="right", style="dim")
    table.add_column("Architecture", style="dim")

    for name, spec in sorted(all_gpus.items(), key=lambda x: -x[1].vram_gb):
        table.add_row(
            name,
            f"{spec.vram_gb:.0f} GB",
            f"{spec.fp16_tflops:.1f}",
            f"{spec.bf16_tflops:.1f}",
            spec.architecture,
        )

    console.print(table)


@app.command()
def pricing(
    gpu: str = typer.Option("A100-80GB", "--gpu", help="GPU to check pricing for"),
    hours: float = typer.Option(10.0, "--hours", "-h", help="Training hours to estimate"),
):
    """Show cloud pricing for a specific GPU."""
    from ftune.cli.display import console
    from ftune.core.cost import CostEstimator

    ce = CostEstimator()
    estimates = ce.estimate_for_gpu(gpu, training_hours=hours)

    if not estimates:
        console.print(f"[red]No pricing data found for GPU '{gpu}'[/red]")
        raise typer.Exit(1)

    from rich.table import Table
    table = Table(title=f"Pricing: {gpu} × {hours}h", border_style="dim")
    table.add_column("Provider", style="white")
    table.add_column("$/hr", justify="right", style="yellow")
    table.add_column("Total", justify="right", style="bold green")
    table.add_column("Spot $/hr", justify="right", style="cyan")
    table.add_column("Spot Total", justify="right", style="cyan")
    table.add_column("Instance", style="dim")

    for e in estimates:
        table.add_row(
            e.provider,
            f"${e.hourly_rate:.2f}",
            f"${e.total_cost:.2f}",
            f"${e.spot_hourly_rate:.2f}" if e.spot_hourly_rate else "—",
            f"${e.spot_total_cost:.2f}" if e.spot_total_cost else "—",
            e.instance_type or "—",
        )

    console.print(table)


@app.command()
def validate(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    method: str = typer.Option("qlora", "--method", help="Fine-tuning method"),
    quantization: str = typer.Option("4bit", "--quantization", "-q", help="Quantization"),
    metrics_file: str = typer.Option(..., "--metrics", "-f", help="Path to JSON metrics file"),
    batch_size: int = typer.Option(4, "--batch-size", "-b"),
    seq_length: int = typer.Option(2048, "--seq-length", "-s"),
    lora_rank: int = typer.Option(16, "--lora-rank", "-r"),
):
    """Validate estimates against actual training metrics."""
    from ftune.cli.display import console
    from ftune.validation import Validator

    if method == "full":
        quantization = "none"

    est = Estimator(
        model=model, method=method, quantization=quantization,
        batch_size=batch_size, seq_length=seq_length, lora_rank=lora_rank,
    )

    actual = Validator.from_json(metrics_file)
    result = Validator.compare(est, actual)
    report = Validator.format_report(result)
    console.print(report)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
