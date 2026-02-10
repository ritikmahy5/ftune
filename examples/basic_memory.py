"""Basic memory estimation example.

Run: python examples/basic_memory.py
"""

from ftune import Estimator, list_model_names, list_gpu_names


def main():
    print("=" * 60)
    print("ftune — LLM Fine-Tuning Memory Estimator")
    print("=" * 60)

    # Available models
    print(f"\nAvailable models ({len(list_model_names())}):")
    for name in list_model_names():
        print(f"  • {name}")

    # Available GPUs
    print(f"\nAvailable GPUs ({len(list_gpu_names())}):")
    for name in list_gpu_names():
        print(f"  • {name}")

    # ── Example 1: QLoRA on Llama 8B ──
    print("\n" + "─" * 60)
    print("Example 1: QLoRA on Llama 3.1 8B")
    print("─" * 60)

    est = Estimator(
        model="meta-llama/Llama-3.1-8B",
        method="qlora",
        quantization="4bit",
        lora_rank=16,
        batch_size=4,
        seq_length=2048,
    )
    print(est.summary())

    # ── Example 2: Full fine-tune on Llama 8B ──
    print("\n" + "─" * 60)
    print("Example 2: Full fine-tuning Llama 3.1 8B")
    print("─" * 60)

    est_full = Estimator(
        model="meta-llama/Llama-3.1-8B",
        method="full",
        batch_size=1,
        seq_length=2048,
    )
    print(est_full.summary())

    # ── Example 3: QLoRA on 70B ──
    print("\n" + "─" * 60)
    print("Example 3: QLoRA on Llama 3.1 70B")
    print("─" * 60)

    est_70b = Estimator(
        model="meta-llama/Llama-3.1-70B",
        method="qlora",
        quantization="4bit",
        lora_rank=16,
        batch_size=1,
        seq_length=2048,
    )
    print(est_70b.summary())


if __name__ == "__main__":
    main()
