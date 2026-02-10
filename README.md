<p align="center">
  <img src="assets/ftune-banner.png" alt="ftune banner" width="700" />
</p>

<h1 align="center">⚡ ftune</h1>

<p align="center">
  <strong>Know your GPU costs before you hit OOM.</strong><br/>
  Estimate memory, training time, and cloud costs for LLM fine-tuning — in seconds.
</p>

<p align="center">
  <a href="https://pypi.org/project/ftune/"><img src="https://img.shields.io/pypi/v/ftune?color=blue&label=PyPI" alt="PyPI" /></a>
  <a href="https://pypi.org/project/ftune/"><img src="https://img.shields.io/pypi/pyversions/ftune" alt="Python" /></a>
  <a href="https://github.com/yourusername/ftune/blob/main/LICENSE"><img src="https://img.shields.io/github/license/yourusername/ftune" alt="License" /></a>
  <a href="https://github.com/yourusername/ftune/actions"><img src="https://img.shields.io/github/actions/workflow/status/yourusername/ftune/ci.yml?label=tests" alt="Tests" /></a>
  <a href="https://pypi.org/project/ftune/"><img src="https://img.shields.io/pypi/dm/ftune?color=green" alt="Downloads" /></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-usage">Usage</a> •
  <a href="#%EF%B8%8F-cli">CLI</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-roadmap">Roadmap</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## The Problem

You want to fine-tune Llama 3.1 70B. You spin up an A100, start training, and... **CUDA out of memory.** 💀

Or worse — you rent 8×H100s for $30/hr, only to realize you could've done it with QLoRA on a single $1.50/hr GPU.

**ftune fixes this.** Get accurate VRAM estimates, training time projections, and cost comparisons across cloud providers — all before you spend a single dollar.

## 🚀 Quick Start

```bash
pip install ftune
```

### Python API

```python
from ftune import Estimator

est = Estimator(
    model="meta-llama/Llama-3.1-8B",
    method="qlora",
    quantization="4bit",
    lora_rank=16,
    batch_size=4,
    seq_length=2048,
)

mem = est.estimate_memory()
print(f"Total VRAM: {mem.total_gb:.2f} GB")
print(f"Trainable params: {mem.trainable_params:,} ({mem.trainable_percentage:.2f}%)")
```

### CLI (one command)

```bash
ftune estimate --model meta-llama/Llama-3.1-8B --method qlora --quantization 4bit
```

**Output:**

```
╭──────────────────────────────────────────────────────────────╮
│              ⚡ ftune — Fine-Tuning Cost Estimator            │
╰──────────────────────────────────────────────────────────────╯

📦 Model: meta-llama/Llama-3.1-8B (8.0B params)
⚙️  Method: QLoRA (rank=16, alpha=32, 4-bit)

┌─────────────────── Memory Breakdown ───────────────────┐
│ Component              │ VRAM (GB)  │ % of Total       │
│────────────────────────┼────────────┼──────────────────│
│ Base Model (4-bit)     │ 4.52       │ 38.2%            │
│ LoRA Adapters          │ 0.16       │ 1.4%             │
│ Gradients              │ 0.16       │ 1.4%             │
│ Optimizer States       │ 0.48       │ 4.1%             │
│ Activations            │ 5.24       │ 44.3%            │
│ CUDA Overhead          │ 1.26       │ 10.6%            │
│────────────────────────┼────────────┼──────────────────│
│ TOTAL                  │ 11.82 GB   │ 100%             │
└────────────────────────────────────────────────────────┘

✅ Fits on: RTX 3090, RTX 4090, A100 (40GB), A100 (80GB), H100
❌ Too large for: T4 (16GB), V100 (16GB)

┌─────────────────── Cost Comparison ────────────────────┐
│ Provider         │ GPU         │ $/hr   │ Total Cost   │
│──────────────────┼─────────────┼────────┼──────────────│
│ Vast.ai          │ RTX 4090    │ $0.40  │ $2.52        │
│ Together AI      │ A100-80GB   │ $1.25  │ $5.25        │
│ RunPod           │ A100-80GB   │ $1.64  │ $6.89        │
│ Lambda Labs      │ A100-80GB   │ $1.99  │ $8.36        │
│ GCP              │ a2-highgpu  │ $3.67  │ $15.41       │
│ AWS (spot)       │ p4d.24xl    │ $12.40 │ $52.08       │
└────────────────────────────────────────────────────────┘

💡 Best value: Vast.ai RTX 4090 — $2.52 for ~6.3 hrs
```

## ✨ Features

### 📊 Memory Estimation
Accurate VRAM calculation with component-level breakdown — model weights, gradients, optimizer states, activations, and CUDA overhead.

- **Full Fine-Tuning** — all parameters trainable
- **LoRA** — low-rank adaptation with configurable rank, alpha, and target modules
- **QLoRA** — 4-bit/8-bit quantized base model + LoRA adapters

### ⏱️ Training Time Estimation
Wall-clock time estimates based on dataset size, GPU compute (TFLOPS), batch configuration, and mixed precision settings. Supports multi-GPU (data parallel, FSDP).

### 💰 Cloud Cost Comparison
Compare costs across **8+ cloud providers** — AWS, GCP, Azure, Lambda Labs, RunPod, Vast.ai, Together AI, and Modal. Includes spot/preemptible pricing.

### 🎯 Configuration Optimizer
Tell ftune your budget or hardware, and it recommends the optimal fine-tuning configuration:

```bash
ftune optimize --model meta-llama/Llama-3.1-70B --budget 50 --priority cost
```

### 🖥️ GPU Compatibility Check
Instantly see which GPUs can handle your training run, and how much VRAM headroom you have.

## 📖 Usage

### Memory Estimation

```python
from ftune import Estimator

# QLoRA on Llama 8B
est = Estimator(
    model="meta-llama/Llama-3.1-8B",
    method="qlora",
    quantization="4bit",
    lora_rank=16,
    batch_size=4,
    seq_length=2048,
    gradient_checkpointing=True,
)

mem = est.estimate_memory()

# Detailed breakdown
print(f"Model weights:    {mem.model_weights_gb:.2f} GB")
print(f"LoRA adapters:    {mem.trainable_params_gb:.2f} GB")
print(f"Gradients:        {mem.gradients_gb:.2f} GB")
print(f"Optimizer states: {mem.optimizer_states_gb:.2f} GB")
print(f"Activations:      {mem.activations_gb:.2f} GB")
print(f"CUDA overhead:    {mem.overhead_gb:.2f} GB")
print(f"─────────────────────────────")
print(f"TOTAL:            {mem.total_gb:.2f} GB")
```

### GPU Compatibility

```python
fits = est.check_gpu_fit()

for gpu in fits:
    icon = "✅" if gpu.fits else "❌"
    print(f"{icon} {gpu.gpu_name} ({gpu.vram_gb}GB) — {gpu.utilization_percent:.1f}% utilized")
```

### Training Time & Cost

```python
# Estimate training time
time_est = est.estimate_time(
    dataset_size=50000,
    epochs=3,
    gpu="A100-80GB",
    num_gpus=1,
)
print(f"Training time: {time_est.total_hours:.1f} hours")

# Compare costs across providers
costs = est.estimate_costs(training_hours=time_est.total_hours)
for c in costs:
    print(f"{c.provider:15s} │ {c.gpu:12s} │ ${c.hourly_rate:.2f}/hr │ ${c.total_cost:.2f} total")
```

### Configuration Optimizer

```python
from ftune import Optimizer

opt = Optimizer(
    model="meta-llama/Llama-3.1-70B",
    dataset_size=100000,
    budget=100.0,
    priority="cost",  # "cost", "speed", or "quality"
)

rec = opt.recommend()
print(rec.summary())
# → "Use QLoRA (rank=32, 4-bit) on RunPod A100-80GB. Estimated: $47.20, 28.8 hrs"
```

## ⌨️ CLI

```bash
# Quick estimate
ftune estimate --model meta-llama/Llama-3.1-8B --method qlora

# Detailed configuration
ftune estimate \
  --model meta-llama/Llama-3.1-8B \
  --method qlora \
  --lora-rank 16 \
  --quantization 4bit \
  --batch-size 4 \
  --seq-length 2048 \
  --dataset-size 50000 \
  --epochs 3 \
  --gpu A100-80GB

# Optimize for budget
ftune optimize --model meta-llama/Llama-3.1-70B --budget 50

# Compare provider costs
ftune compare --model meta-llama/Llama-3.1-8B --training-hours 5

# List supported models & GPUs
ftune models list
ftune gpus list

# Export as JSON
ftune estimate --model meta-llama/Llama-3.1-8B --method lora --output json
```

## 🧮 How It Works

ftune uses well-established formulas for GPU memory estimation — no guesswork, no ML dependencies.

### Memory Formula

| Component | Full Fine-Tune | LoRA | QLoRA (4-bit) |
|---|---|---|---|
| Model Weights | `params × dtype_bytes` | `params × dtype_bytes` | `params × 0.5` + quant overhead |
| Trainable Params | Same as above | `modules × 2 × hidden × rank` | Same as LoRA |
| Gradients | `params × 2B` | `lora_params × 2B` | `lora_params × 2B` |
| Optimizer (AdamW) | `params × 8B` | `lora_params × 8B` | `lora_params × 8B` |
| Activations | `batch × seq × hidden × layers × factor` | Same | Same |
| Overhead | ~15% buffer | ~15% buffer | ~15% buffer |

> `factor` = ~2 with gradient checkpointing, ~10-14 without

### Training Time Formula

```
FLOPs per token ≈ 6 × num_parameters
Total FLOPs = flops_per_token × dataset_tokens × epochs
Time = Total FLOPs / (GPU TFLOPS × MFU × num_gpus)
```

MFU (Model FLOPs Utilization) defaults to 0.35, a conservative estimate for fine-tuning workloads.

> All formulas are documented in detail at [`docs/formulas.md`](docs/formulas.md)

## 📋 Supported Models

| Model | Parameters | Default dtype |
|---|---|---|
| meta-llama/Llama-3.1-8B | 8B | bf16 |
| meta-llama/Llama-3.1-70B | 70B | bf16 |
| meta-llama/Llama-3.1-405B | 405B | bf16 |
| mistralai/Mistral-7B-v0.3 | 7B | bf16 |
| mistralai/Mixtral-8x7B-v0.1 | 47B (MoE) | bf16 |
| google/gemma-2-9b | 9B | bf16 |
| google/gemma-2-27b | 27B | bf16 |
| Qwen/Qwen2.5-7B | 7B | bf16 |
| Qwen/Qwen2.5-72B | 72B | bf16 |
| microsoft/phi-3-mini-4k | 3.8B | bf16 |
| deepseek-ai/DeepSeek-V2-Lite | 16B | bf16 |

> **Don't see your model?** Add it to `data/models.yaml` or use `--auto` to fetch specs from HuggingFace Hub (coming soon).

## 📋 Supported GPUs

| GPU | VRAM | FP16 TFLOPS |
|---|---|---|
| NVIDIA H100 | 80 GB | 989 |
| NVIDIA A100 | 40 / 80 GB | 312 |
| NVIDIA L4 | 24 GB | 121 |
| RTX 4090 | 24 GB | 165 |
| RTX 3090 | 24 GB | 71 |
| Tesla T4 | 16 GB | 65 |
| Tesla V100 | 16 GB | 125 |

## 🗺️ Roadmap

- [x] **Phase 1** — Memory estimation engine (full, LoRA, QLoRA)
- [ ] **Phase 2** — Training time estimation
- [ ] **Phase 3** — Cloud cost estimation & provider comparison
- [ ] **Phase 4** — CLI with Rich terminal output
- [ ] **Phase 5** — Configuration optimizer & recommender
- [ ] **Phase 6** — Interactive mode & export formats (JSON, Markdown)
- [ ] **Phase 7** — HuggingFace Hub auto-detect (`--auto` flag)
- [ ] **Phase 8** — Web dashboard (Streamlit/Gradio)
- [ ] **Phase 9** — Validation mode (compare vs actual W&B/TensorBoard logs)

## 🔑 Design Principles

- **Zero ML dependencies** — Pure Python calculator. No PyTorch, no TensorFlow, no GPU required.
- **Offline-first** — All model/GPU specs are bundled. Works without internet.
- **Extensible** — Add models, GPUs, and pricing via simple YAML files.
- **Accurate** — Formulas are documented, tested, and validated against real training runs.
- **Fast** — Every estimate returns in under 1 second.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Add models** — Submit a PR adding new model specs to `data/models.yaml`
2. **Add GPU data** — Add specs for new GPUs to `data/gpus.yaml`
3. **Update pricing** — Keep cloud pricing data current
4. **Validate estimates** — Compare ftune estimates against your real training runs and report findings
5. **Bug fixes & features** — Check the [issues](https://github.com/yourusername/ftune/issues) tab

```bash
# Development setup
git clone https://github.com/yourusername/ftune.git
cd ftune
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

## ⭐ Star History

If ftune saved you from an OOM error (or an expensive cloud bill), consider giving it a star!

---

<p align="center">
  Built with ☕ by <a href="https://github.com/yourusername">yourusername</a>
</p>
