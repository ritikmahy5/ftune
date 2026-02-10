<p align="center">
  <h1 align="center">⚡ ftune</h1>
</p>

<p align="center">
  <strong>Know your GPU costs before you hit OOM.</strong><br/>
  Estimate memory, training time, and cloud costs for LLM fine-tuning — in seconds.
</p>

<p align="center">
  <a href="https://pypi.org/project/ftune/"><img src="https://img.shields.io/pypi/v/ftune?color=blue&label=PyPI" alt="PyPI" /></a>
  <a href="https://pypi.org/project/ftune/"><img src="https://img.shields.io/pypi/pyversions/ftune" alt="Python" /></a>
  <a href="https://github.com/ritikmahy5/ftune/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ritikmahy5/ftune" alt="License" /></a>
  <a href="https://github.com/ritikmahy5/ftune/actions"><img src="https://img.shields.io/github/actions/workflow/status/ritikmahy5/ftune/ci.yml?label=tests" alt="Tests" /></a>
</p>

<p align="center">
  <a href="#-the-problem">Problem</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-web-ui">Web UI</a> •
  <a href="#-features">Features</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-validation">Validation</a> •
  <a href="#-supported-models--gpus">Models & GPUs</a> •
  <a href="#-roadmap">Roadmap</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🔥 The Problem

You want to fine-tune Llama 3.1 70B. You spin up an A100, start training, and... **CUDA out of memory** 💀

Or worse — you rent 8×H100s for $30/hr, only to realize you could've done it with QLoRA on a single $1.50/hr GPU.

**ftune fixes this.** Get accurate VRAM estimates, training time projections, and cost comparisons across 8 cloud providers — all before you spend a single dollar.

### What makes ftune different?

- **Works with ANY model** — fetches specs from HuggingFace Hub automatically, not just a hardcoded list
- **Full cost comparison** — compares 8 cloud providers including spot pricing in one command
- **Validation mode** — compare estimates against your actual W&B/TensorBoard logs to see how accurate they are
- **Interactive web UI** — no install needed, just point and click
- **Zero ML dependencies** — pure Python calculator, no PyTorch/TensorFlow required

---

## 🚀 Quick Start

```bash
pip install ftune
```

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

# Memory
mem = est.estimate_memory()
print(f"Total VRAM: {mem.total_gb:.2f} GB")

# Training time
time = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3)
print(f"Training time: {time.total_hours:.1f} hours")

# Cost comparison across all providers
costs = est.full_comparison(dataset_size=50000, epochs=3)
for c in costs.estimates[:5]:
    print(f"{c.provider:15s} | {c.gpu:15s} | ${c.total_cost:.2f}")
```

**Output:**

```
Total VRAM: 9.09 GB
Training time: 32.9 hours

Vast.ai         | RTX-4090-24GB   | $24.89
Lambda Labs     | H100-80GB       | $25.87
Together AI     | H100-80GB       | $25.98
Vast.ai         | H100-80GB       | $29.61
Vast.ai         | A100-40GB       | $29.65
```

### Works with ANY HuggingFace model

Don't see your model in the built-in database? ftune auto-fetches specs from HuggingFace Hub:

```python
# Any model on HuggingFace — no configuration needed
est = Estimator(model="NousResearch/Llama-2-7b-hf", method="qlora", quantization="4bit")
est = Estimator(model="tiiuae/falcon-40b", method="lora", lora_rank=32)
est = Estimator(model="bigscience/bloom-7b1", method="qlora", quantization="4bit")
```

---

## 🌐 Web UI

ftune includes a full interactive web calculator built with Streamlit. No Python knowledge needed.

```bash
pip install ftune[web]
streamlit run src/ftune/app.py
```

**Features:**
- Sidebar with all configuration options (model, method, LoRA rank, batch size, etc.)
- 📊 **Memory tab** — VRAM breakdown with bar chart
- ⏱️ **Training Time tab** — time estimates across all compatible GPUs
- 💰 **Cost tab** — full provider comparison with spot pricing
- 🖥️ **GPU Compatibility tab** — utilization chart showing which GPUs fit

<!-- > 🔗 **Try it live:** [ftune.streamlit.app](https://ftune.streamlit.app) *(coming soon)* -->

---

## ✨ Features

### 📊 Memory Estimation

Accurate VRAM calculation with component-level breakdown — model weights, gradients, optimizer states, activations, and CUDA overhead.

| Method | What it does |
|---|---|
| **Full Fine-Tuning** | All parameters trainable, highest memory |
| **LoRA** | Low-rank adapters, base model in fp16/bf16 |
| **QLoRA** | 4-bit/8-bit quantized base + LoRA adapters |

```python
mem = est.estimate_memory()

print(f"Model weights:    {mem.model_weights_gb:.2f} GB")
print(f"LoRA adapters:    {mem.trainable_params_gb:.2f} GB")
print(f"Gradients:        {mem.gradients_gb:.2f} GB")
print(f"Optimizer states: {mem.optimizer_states_gb:.2f} GB")
print(f"Activations:      {mem.activations_gb:.2f} GB")
print(f"CUDA overhead:    {mem.overhead_gb:.2f} GB")
print(f"TOTAL:            {mem.total_gb:.2f} GB")
```

### ⏱️ Training Time Estimation

Wall-clock time estimates based on FLOPs calculation, GPU compute capacity, and Model FLOPs Utilization (MFU).

```python
# Single GPU
time = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3)
print(f"{time.total_hours:.1f} hours, {time.total_steps:,} steps")

# Compare across all compatible GPUs
for t in est.estimate_time_all_gpus(dataset_size=50000, epochs=3):
    print(f"{t.gpu_name:<18} {t.total_hours:>6.1f}h")
```

Supports multi-GPU estimation with realistic scaling efficiency (accounts for communication overhead).

### 💰 Cloud Cost Comparison

Compare costs across **8 cloud providers** with a single call:

```python
costs = est.full_comparison(dataset_size=50000, epochs=3)

for c in costs.estimates[:10]:
    spot = f"${c.spot_total_cost:.2f}" if c.spot_total_cost else "—"
    print(f"{c.provider:<18} {c.gpu:<16} ${c.total_cost:>8.2f}  spot: {spot}")

print(f"\n🏆 Cheapest: {costs.cheapest}")
print(f"💡 Best value: {costs.best_value}")
```

**Supported providers:** AWS, Google Cloud, Microsoft Azure, Lambda Labs, RunPod, Vast.ai, Together AI, Modal — with on-demand and spot pricing.

### 🖥️ GPU Compatibility Check

Instantly see which GPUs can handle your training run:

```python
for gpu in est.check_gpu_fit():
    icon = "✅" if gpu.fits else "❌"
    print(f"{icon} {gpu.gpu_name:<18} {gpu.vram_gb:>4.0f}GB  {gpu.utilization_percent:>5.1f}% used")
```

```
✅ H100-80GB          80GB   11.4% used
✅ A100-80GB          80GB   11.4% used
✅ A100-40GB          40GB   22.7% used
✅ RTX-4090-24GB      24GB   37.9% used
✅ T4-16GB            16GB   56.8% used
❌ — (none too small for QLoRA 8B!)
```

### 🔬 HuggingFace Hub Auto-Detect

ftune works with **any model on HuggingFace Hub** — not just a hardcoded list. When a model isn't in the local database, ftune automatically fetches `config.json` from HuggingFace and extracts architecture specs:

```python
from ftune import resolve_model_from_hub

# Fetch specs for any model
spec = resolve_model_from_hub("NousResearch/Llama-2-7b-hf")
print(f"{spec.name}: {spec.parameters:,} params, hidden={spec.hidden_size}")
```

This happens automatically when you use `Estimator` with an unknown model name — no extra code needed.

### 📊 Validation Mode

Compare ftune estimates against your **actual training runs** to see how accurate the estimates are:

```python
from ftune import Estimator
from ftune.validation import Validator, ActualMetrics

est = Estimator(model="meta-llama/Llama-3.1-8B", method="qlora", quantization="4bit")

# From your actual training run
actual = ActualMetrics(
    peak_memory_gb=11.2,          # nvidia-smi or torch.cuda.max_memory_allocated()
    training_time_hours=4.5,       # wall-clock time
    total_cost=8.50,               # cloud bill
    gpu_name="A100-80GB",
    dataset_size=50000,
    epochs=3,
)

result = Validator.compare(est, actual)
print(Validator.format_report(result))
```

```
╭────────────────────────────────────────────────╮
│        📊 ftune Validation Report               │
╰────────────────────────────────────────────────╯

Memory:
  Estimated:  9.09 GB
  Actual:     11.20 GB
  ✅ Error:  -2.11 GB (-18.8%)

Training Time:
  Estimated:  32.94 hours
  Actual:     4.50 hours
  ⚠️ Error:  +28.44 hours (+632.0%)

Overall Accuracy: Fair (20-35% avg error)
```

**Load metrics from multiple sources:**

```python
# From a JSON file
actual = Validator.from_json("training_metrics.json")

# From Weights & Biases
actual = Validator.from_wandb("username/project/run_id")  # pip install ftune[wandb]

# From a CSV (batch validation)
metrics_list = Validator.from_csv("all_runs.csv")
```

---

## 🧮 How It Works

ftune uses well-established formulas for GPU memory estimation — no guesswork, no ML dependencies.

### Memory Formula

| Component | Full Fine-Tune | LoRA | QLoRA (4-bit) |
|---|---|---|---|
| Model Weights | `params × dtype_bytes` | `params × dtype_bytes` | `params × 0.5` + quant overhead |
| Trainable Params | (same as weights) | `modules × 2 × hidden × rank` | Same as LoRA |
| Gradients | `params × 2B` | `lora_params × 2B` | `lora_params × 2B` |
| Optimizer (AdamW) | `params × 8B` | `lora_params × 8B` | `lora_params × 8B` |
| Activations | `batch × seq × hidden × layers × factor` | Same | Same |
| Overhead | ~15% buffer | ~15% buffer | ~15% buffer |

> `factor` ≈ 2 with gradient checkpointing, ≈ 10 without

### Training Time Formula

```
FLOPs per token ≈ 6 × num_parameters
Total FLOPs = flops_per_token × dataset_tokens × epochs
Time = Total FLOPs / (GPU TFLOPS × MFU × num_gpus)
```

MFU (Model FLOPs Utilization) defaults to 0.30-0.35, conservative estimates for fine-tuning workloads.

---

## 📋 Supported Models & GPUs

### Built-in Models (15)

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
| microsoft/phi-3-mini-4k-instruct | 3.8B | bf16 |
| microsoft/phi-3-medium-4k-instruct | 14B | bf16 |
| deepseek-ai/DeepSeek-V2-Lite | 16B | bf16 |
| 01-ai/Yi-1.5-9B | 8.8B | bf16 |
| tiiuae/falcon-7b | 6.9B | bf16 |
| stabilityai/stablelm-2-1_6b | 1.6B | bf16 |

> **Plus ANY model on HuggingFace Hub** — ftune auto-fetches specs when a model isn't in the local database.

### Supported GPUs (11)

| GPU | VRAM | FP16 TFLOPS |
|---|---|---|
| NVIDIA H100 | 80 GB | 989 |
| NVIDIA A100 | 40 / 80 GB | 312 |
| NVIDIA A10G | 24 GB | 125 |
| NVIDIA L4 | 24 GB | 121 |
| RTX 4090 | 24 GB | 165 |
| RTX 4080 | 16 GB | 97 |
| RTX 3090 | 24 GB | 71 |
| Tesla T4 | 16 GB | 65 |
| Tesla V100 | 16 / 32 GB | 125 |

### Cloud Providers (8)

| Provider | GPUs Available | Spot Pricing |
|---|---|---|
| AWS | H100, A100, T4, L4 | ✅ |
| Google Cloud | H100, A100, T4, L4, V100 | ✅ |
| Microsoft Azure | H100, A100, T4, V100 | ✅ |
| Lambda Labs | H100, A100, A100-40G, RTX 4090 | — |
| RunPod | H100, A100, A100-40G, RTX 4090, RTX 3090, L4 | ✅ |
| Vast.ai | H100, A100, A100-40G, RTX 4090, RTX 3090 | — |
| Together AI | H100, A100 | — |
| Modal | H100, A100, A100-40G, L4, T4 | — |

---

## 📦 Installation Options

```bash
# Core library only (no extra dependencies)
pip install ftune

# With Streamlit web UI
pip install ftune[web]

# With W&B validation support
pip install ftune[wandb]

# Everything
pip install ftune[all]
```

---

## 🗺️ Roadmap

- [x] Memory estimation engine (full, LoRA, QLoRA)
- [x] Training time estimation (FLOPs-based, multi-GPU)
- [x] Cloud cost comparison (8 providers, spot pricing)
- [x] HuggingFace Hub auto-detect (any model)
- [x] Validation mode (manual, JSON, W&B, CSV)
- [x] Streamlit web UI
- [ ] CLI with Rich terminal output (`ftune estimate --model ...`)
- [ ] Configuration optimizer ("I have $50 budget, what's optimal?")
- [ ] Export reports (JSON, Markdown, PDF)
- [ ] Community validation dataset (crowdsourced accuracy data)
- [ ] Streamlit Cloud deployment (public hosted version)

---

## 🔑 Design Principles

- **Zero ML dependencies** — Pure Python calculator. No PyTorch, no TensorFlow, no GPU required.
- **Works with any model** — HuggingFace Hub integration means instant support for thousands of models.
- **Offline-first** — 15 models and 11 GPUs bundled. Works without internet for common models.
- **Validates itself** — Built-in validation mode so you can see how accurate estimates really are.
- **Fast** — Every estimate returns in under 1 second.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Validate estimates** — Run ftune against your actual training runs and share the results. This is the highest-impact contribution.
2. **Add models** — Submit a PR adding new model specs to `data/models.yaml`
3. **Update pricing** — Cloud pricing changes frequently. Help keep `data/pricing.yaml` current.
4. **Add GPU data** — Add specs for new GPUs to `data/gpus.yaml`
5. **Bug fixes & features** — Check the [issues](https://github.com/ritikmahy5/ftune/issues) tab

```bash
# Development setup
git clone https://github.com/ritikmahy5/ftune.git
cd ftune
pip install -e ".[dev]"

# Run tests
PYTHONPATH=src pytest tests/

# Run the web UI locally
pip install streamlit pandas altair
PYTHONPATH=src streamlit run src/ftune/app.py
```

### Sharing Validation Results

The most valuable contribution is comparing ftune estimates against real training runs:

```python
from ftune import Estimator
from ftune.validation import Validator, ActualMetrics

est = Estimator(model="your-model", method="qlora", quantization="4bit", ...)

actual = ActualMetrics(
    peak_memory_gb=...,       # from nvidia-smi
    training_time_hours=...,   # wall-clock
    gpu_name="...",
    dataset_size=...,
    epochs=...,
)

result = Validator.compare(est, actual)
print(Validator.format_report(result))
# Share this output in an issue or PR!
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>If ftune saved you from an OOM error or an expensive cloud bill, consider giving it a ⭐</strong>
</p>

<p align="center">
  Built by <a href="https://github.com/ritikmahy5">Ritik Mahyavanshi</a>
</p>
