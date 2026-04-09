"""Full estimation example: memory + time + cost.

Run: python examples/full_estimate.py
"""

from ftuneai import Estimator


def main():
    print("=" * 65)
    print("  ⚡ ftune — Full Estimation: Memory + Time + Cost")
    print("=" * 65)

    est = Estimator(
        model="meta-llama/Llama-3.1-8B",
        method="qlora",
        quantization="4bit",
        lora_rank=16,
        batch_size=4,
        seq_length=2048,
    )

    # ── Memory ──
    mem = est.estimate_memory()
    print(f"\n📦 Model: {est.model_spec.name}")
    print(f"⚙️  Method: QLoRA (rank=16, 4-bit)\n")
    print("┌─── Memory Breakdown ────────────────────────┐")
    print(f"│ Model weights:    {mem.model_weights_gb:>8.2f} GB              │")
    print(f"│ LoRA adapters:    {mem.trainable_params_gb:>8.2f} GB              │")
    print(f"│ Gradients:        {mem.gradients_gb:>8.2f} GB              │")
    print(f"│ Optimizer states: {mem.optimizer_states_gb:>8.2f} GB              │")
    print(f"│ Activations:      {mem.activations_gb:>8.2f} GB              │")
    print(f"│ CUDA overhead:    {mem.overhead_gb:>8.2f} GB              │")
    print(f"│─────────────────────────────────────────────│")
    print(f"│ TOTAL:            {mem.total_gb:>8.2f} GB              │")
    print("└─────────────────────────────────────────────┘")

    # ── GPU Compatibility ──
    fits = est.check_gpu_fit()
    compatible = [g for g in fits if g.fits]
    incompatible = [g for g in fits if not g.fits]
    print(f"\n✅ Fits on: {', '.join(g.gpu_name for g in compatible)}")
    if incompatible:
        print(f"❌ Too large for: {', '.join(g.gpu_name for g in incompatible)}")

    # ── Training Time ──
    dataset_size = 50000
    epochs = 3
    print(f"\n⏱️  Training Time (dataset={dataset_size:,}, epochs={epochs}):\n")
    print(f"{'GPU':<18} {'Time/Epoch':>12} {'Total':>12}")
    print("─" * 44)

    times = est.estimate_time_all_gpus(dataset_size=dataset_size, epochs=epochs)
    for t in times[:6]:  # Show top 6
        print(f"{t.gpu_name:<18} {t.hours_per_epoch:>10.1f}h {t.total_hours:>10.1f}h")

    # ── Cost Comparison ──
    print(f"\n💰 Cost Comparison (using A100-80GB, {times[0].total_hours if times else '?'}h):\n")

    best_gpu = times[0].gpu_name if times else "A100-80GB"
    costs = est.estimate_costs(
        gpu="A100-80GB",
        dataset_size=dataset_size,
        epochs=epochs,
    )

    print(f"{'Provider':<18} {'$/hr':>8} {'Total':>10} {'Spot':>10}")
    print("─" * 48)
    for c in costs.estimates:
        spot = f"${c.spot_total_cost:.2f}" if c.spot_total_cost else "—"
        print(f"{c.provider:<18} ${c.hourly_rate:>6.2f} ${c.total_cost:>8.2f} {spot:>10}")

    if costs.cheapest:
        print(f"\n💡 Cheapest on-demand: {costs.cheapest}")
    if costs.best_value:
        print(f"🏆 Best value: {costs.best_value}")

    # ── Full comparison ──
    print("\n" + "=" * 65)
    print("  🔍 Full Comparison (all GPUs × all providers)")
    print("=" * 65 + "\n")

    full = est.full_comparison(dataset_size=dataset_size, epochs=epochs)
    print(f"{'Provider':<18} {'GPU':<16} {'Hours':>6} {'$/hr':>8} {'Total':>10}")
    print("─" * 60)
    for c in full.estimates[:15]:  # Top 15 cheapest
        print(
            f"{c.provider:<18} {c.gpu:<16} {c.training_hours:>5.1f}h "
            f"${c.hourly_rate:>6.2f} ${c.total_cost:>8.2f}"
        )

    print(f"\n🏆 Cheapest option: {full.cheapest}")
    print(f"💡 Best value: {full.best_value}")


if __name__ == "__main__":
    main()
