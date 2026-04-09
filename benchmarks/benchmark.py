"""ftune Validation Benchmark — measure real training metrics vs ftune estimates.

Runs a short fine-tuning job (default 20 steps) and records:
  - Peak GPU VRAM (nvidia-smi)
  - Wall-clock time per step
  - Extrapolated total training time

Then compares against ftune's analytical estimates.

Usage:
    python benchmark.py --model meta-llama/Llama-3.1-8B --method qlora --gpu V100-16GB
    python benchmark.py --model microsoft/phi-3-mini-4k-instruct --method lora --gpu A100-80GB

Requirements:
    pip install torch transformers peft bitsandbytes datasets accelerate ftuneai
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset


def get_gpu_name() -> str:
    """Get GPU name from torch."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "unknown"


def get_peak_memory_mb() -> float:
    """Get peak GPU memory from nvidia-smi in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip().split("\n")[0])
    except Exception:
        return torch.cuda.max_memory_allocated() / 1024 / 1024


def get_peak_memory_gb() -> float:
    """Get peak allocated GPU memory in GB from PyTorch."""
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def create_dummy_dataset(tokenizer, num_samples: int = 200, seq_length: int = 512) -> Dataset:
    """Create a dummy training dataset."""
    texts = [
        "This is a sample training text for benchmarking fine-tuning performance. "
        "The quick brown fox jumps over the lazy dog. " * (seq_length // 20)
    ] * num_samples

    encodings = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=seq_length, return_tensors="pt",
    )

    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["input_ids"].clone(),
    })


def run_benchmark(
    model_name: str,
    method: str = "qlora",
    quantization: str = "4bit",
    lora_rank: int = 16,
    batch_size: int = 1,
    seq_length: int = 512,
    num_steps: int = 20,
    dataset_size: int = 200,
    lora_target: str = "attention",
    flash_attention: bool = False,
) -> dict:
    """Run a short training benchmark and return metrics."""

    fa_str = " | FlashAttention-2" if flash_attention else ""
    print(f"\n{'='*60}")
    print(f"Benchmark: {model_name} | {method} | {quantization}{fa_str}")
    print(f"GPU: {get_gpu_name()}")
    print(f"Steps: {num_steps} | Batch: {batch_size} | Seq: {seq_length}")
    print(f"{'='*60}\n")

    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Load model
    print("Loading model...")
    load_kwargs = {}
    if method == "qlora":
        from transformers import BitsAndBytesConfig
        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    attn_impl = "flash_attention_2" if flash_attention else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if method != "qlora" else None,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
        **load_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA if applicable
    trainable_params = 0
    total_params = sum(p.numel() for p in model.parameters())

    if method in ("lora", "qlora"):
        if method == "qlora":
            model = prepare_model_for_kbit_training(model)

        # Map lora_target to module names
        if lora_target == "attention":
            target_modules = ["q_proj", "v_proj"]
        elif lora_target == "attention_all":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif lora_target == "all_linear":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} "
          f"({100 * trainable_params / total_params:.4f}%)")

    # Create dataset
    print("Creating dataset...")
    dataset = create_dummy_dataset(tokenizer, num_samples=dataset_size, seq_length=seq_length)

    # Training
    output_dir = f"/tmp/ftune_benchmark_{int(time.time())}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=num_steps,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        dataloader_pin_memory=False,
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"\nTraining {num_steps} steps...")
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # Collect metrics
    peak_memory_gb = get_peak_memory_gb()
    seconds_per_step = elapsed / num_steps
    tokens_per_step = batch_size * seq_length
    tokens_per_second = tokens_per_step / seconds_per_step

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Peak VRAM:          {peak_memory_gb:.2f} GB")
    print(f"  Wall-clock time:    {elapsed:.1f}s ({num_steps} steps)")
    print(f"  Seconds/step:       {seconds_per_step:.2f}")
    print(f"  Tokens/second:      {tokens_per_second:.1f}")
    print(f"  Trainable params:   {trainable_params:,}")
    print(f"{'='*60}\n")

    # Clean up
    del model, trainer
    torch.cuda.empty_cache()

    # Remove output dir
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)

    return {
        "model": model_name,
        "method": method,
        "quantization": quantization,
        "lora_rank": lora_rank,
        "lora_target": lora_target,
        "flash_attention": flash_attention,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "num_steps": num_steps,
        "dataset_size": dataset_size,
        "gpu_name": get_gpu_name(),
        "peak_memory_gb": round(peak_memory_gb, 2),
        "wall_clock_seconds": round(elapsed, 2),
        "seconds_per_step": round(seconds_per_step, 3),
        "tokens_per_second": round(tokens_per_second, 1),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "timestamp": datetime.now().isoformat(),
    }


def compare_with_ftune(result: dict, ftune_gpu: str) -> dict:
    """Compare benchmark results against ftune estimates."""
    from ftuneai import Estimator

    est = Estimator(
        model=result["model"],
        method=result["method"],
        quantization=result["quantization"],
        lora_rank=result["lora_rank"],
        lora_target=result["lora_target"],
        flash_attention=result.get("flash_attention", False),
        batch_size=result["batch_size"],
        seq_length=result["seq_length"],
        gradient_checkpointing=True,
    )

    mem = est.estimate_memory()

    # Extrapolate time for a typical run (50k samples, 3 epochs)
    dataset_size = 50000
    epochs = 3
    time_est = est.estimate_time(gpu=ftune_gpu, dataset_size=dataset_size, epochs=epochs)

    # Extrapolate actual time from benchmark
    total_steps_full = (dataset_size * epochs) // result["batch_size"]
    actual_hours = (result["seconds_per_step"] * total_steps_full) / 3600

    mem_error_pct = ((mem.total_gb - result["peak_memory_gb"]) / result["peak_memory_gb"]) * 100
    time_error_pct = ((time_est.total_hours - actual_hours) / actual_hours) * 100

    comparison = {
        "ftune_memory_gb": mem.total_gb,
        "actual_memory_gb": result["peak_memory_gb"],
        "memory_error_pct": round(mem_error_pct, 1),
        "ftune_time_hours": time_est.total_hours,
        "actual_time_hours_extrapolated": round(actual_hours, 2),
        "time_error_pct": round(time_error_pct, 1),
        "ftune_trainable_params": mem.trainable_params,
        "actual_trainable_params": result["trainable_params"],
        "extrapolation_basis": f"{result['num_steps']} steps -> {total_steps_full} steps "
                               f"({dataset_size} samples x {epochs} epochs)",
    }

    print(f"\n{'='*60}")
    print(f"ftune vs Actual Comparison")
    print(f"{'='*60}")
    print(f"  Memory:  ftune={mem.total_gb:.2f} GB  actual={result['peak_memory_gb']:.2f} GB  "
          f"error={mem_error_pct:+.1f}%")
    print(f"  Time:    ftune={time_est.total_hours:.2f} hr  actual~={actual_hours:.2f} hr  "
          f"error={time_error_pct:+.1f}%")
    print(f"  Params:  ftune={mem.trainable_params:,}  actual={result['trainable_params']:,}")
    print(f"{'='*60}\n")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="ftune Validation Benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--method", default="qlora", choices=["full", "lora", "qlora"])
    parser.add_argument("--quantization", default="4bit", choices=["none", "4bit", "8bit"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-target", default="attention",
                        choices=["attention", "attention_all", "all_linear"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--dataset-size", type=int, default=200)
    parser.add_argument("--flash-attention", action="store_true",
                        help="Enable FlashAttention-2")
    parser.add_argument("--gpu", required=True,
                        help="ftune GPU name for comparison (e.g. V100-16GB, A100-80GB)")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    if args.method == "full":
        args.quantization = "none"

    result = run_benchmark(
        model_name=args.model,
        method=args.method,
        quantization=args.quantization,
        lora_rank=args.lora_rank,
        lora_target=args.lora_target,
        flash_attention=args.flash_attention,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_steps=args.num_steps,
        dataset_size=args.dataset_size,
    )

    comparison = compare_with_ftune(result, ftune_gpu=args.gpu)

    # Combine and save
    output = {**result, "comparison": comparison}
    output_path = Path(args.output)

    # Append to existing results if file exists
    results = []
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        if not isinstance(results, list):
            results = [results]

    results.append(output)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
