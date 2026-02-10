"""Validation example — compare ftune estimates against actual training metrics.

Run: python examples/validate.py
"""

from ftune import Estimator
from ftune.validation import Validator, ActualMetrics


def main():
    print("=" * 60)
    print("  📊 ftune Validation — Estimate vs Reality")
    print("=" * 60)

    # ── Example 1: Manual validation ──
    print("\n── Example 1: QLoRA on Llama 8B (A100-80GB) ──\n")

    est = Estimator(
        model="meta-llama/Llama-3.1-8B",
        method="qlora",
        quantization="4bit",
        lora_rank=16,
        batch_size=4,
        seq_length=2048,
    )

    # These would come from your actual training run
    actual = ActualMetrics(
        peak_memory_gb=11.2,          # From nvidia-smi or torch.cuda.max_memory_allocated()
        training_time_hours=4.5,       # Actual wall-clock time
        total_cost=8.50,               # What you actually paid
        gpu_name="A100-80GB",
        num_gpus=1,
        dataset_size=50000,
        epochs=3,
    )

    result = Validator.compare(est, actual)
    print(Validator.format_report(result))

    # ── Example 2: Full fine-tune validation ──
    print("\n── Example 2: Full fine-tune Llama 8B (H100) ──\n")

    est_full = Estimator(
        model="meta-llama/Llama-3.1-8B",
        method="full",
        batch_size=2,
        seq_length=2048,
    )

    actual_full = ActualMetrics(
        peak_memory_gb=95.0,
        training_time_hours=8.0,
        gpu_name="H100-80GB",
        num_gpus=2,
        dataset_size=100000,
        epochs=1,
    )

    result_full = Validator.compare(est_full, actual_full)
    print(Validator.format_report(result_full))

    # ── Example 3: Load from JSON ──
    print("\n── Example 3: Loading metrics from JSON ──\n")
    print("To validate from a file, save your metrics as JSON:")
    print("""
    {
        "peak_memory_gb": 11.2,
        "training_time_hours": 4.5,
        "total_cost": 8.50,
        "gpu_name": "A100-80GB",
        "num_gpus": 1,
        "dataset_size": 50000,
        "epochs": 3,
        "model_name": "meta-llama/Llama-3.1-8B",
        "method": "qlora"
    }
    """)
    print("Then: actual = Validator.from_json('metrics.json')")
    print("\nFor W&B integration:")
    print("  actual = Validator.from_wandb('username/project/run_id')")


if __name__ == "__main__":
    main()
