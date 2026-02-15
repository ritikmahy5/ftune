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
  <a href="#-budget-optimizer">Budget Optimizer</a> •
  <a href="#-multi-gpu--sharding">Multi-GPU</a> •
  <a href="#-calibration">Calibration</a> •
  <a href="#-validation">Validation</a> •
  <a href="#-cli">CLI</a> •
  <a href="#-supported-models-gpus--providers">Models & GPUs</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🔥 The Problem

You want to fine-tune Llama 3.1 70B. You spin up an A100, start training, and... **CUDA out of memory** 💀

Or worse — you rent 8×H100s for $30/hr, only to realize you could've done it with QLoRA on a single $1.50/hr GPU.

**ftune fixes this.** Get accurate VRAM estimates, training time projections, and cost comparisons across 8 cloud providers — all before you spend a single dollar.

### What makes ftune different?

| Feature | ftune | Manual math | HF `accelerate estimate` |
|---|---|---|---|
| Works with any HuggingFace model | ✅ auto-fetches from Hub | ❌ | ✅ |
| Multi-GPU / ZeRO / FSDP sharding | ✅ ZeRO 1/2/3 + FSDP | ❌ | ❌ |
| Cloud cost comparison (8 providers) | ✅ with spot pricing | ❌ | ❌ |
| Budget optimizer ("I have $50") | ✅ | ❌ | ❌ |
| Hardware calibration mode | ✅ | ❌ | ❌ |
| FlashAttention-2 memory savings | ✅ | manual | ❌ |
| Validation against real runs | ✅ W&B, JSON, CSV | ❌ | ❌ |
| Zero ML dependencies | ✅ pure Python | ✅ | ❌ needs torch |

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
    flash_attention=True,
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

### Works with ANY HuggingFace model

ftune auto-fetches model architecture from HuggingFace Hub — no configuration needed:

```python
est = Estimator(model="NousResearch/Llama-2-7b-hf", method="qlora", quantization="4bit")
est = Estimator(model="tiiuae/falcon-40b", method="lora", lora_rank=32)
est = Estimator(model="bigscience/bloom-7b1", method="qlora", quantization="4bit")
```

---

## 🌐 Web UI

ftune includes a full interactive web calculator built with Streamlit.

```bash
pip install ftune[web]
streamlit run src/ftune/app.py
```

Four tabs: **Memory** (VRAM breakdown + chart), **Training Time** (all GPUs), **Cost** (provider comparison + spot pricing), **GPU Compatibility** (utilization chart).

> 🔗 **Try it live:** [ftuneai.streamlit.app](https://ftuneai.streamlit.app)

---

## ✨ Features

### 📊 Memory Estimation

Component-level VRAM breakdown with support for FlashAttention-2, gradient checkpointing, and 5 optimizer types:

```python
est = Estimator(
    model="meta-llama/Llama-3.1-8B",
    method="qlora",
    quantization="4bit",
    flash_attention=True,        # 25-50% activation memory reduction
    gradient_checkpointing=True, # 5x activation reduction
    optimizer="adam_8bit",       # 75% less optimizer memory vs AdamW
)

mem = est.estimate_memory()
print(f"Model weights:    {mem.model_weights_gb:.2f} GB")
print(f"LoRA adapters:    {mem.trainable_params_gb:.2f} GB")
print(f"Gradients:        {mem.gradients_gb:.2f} GB")
print(f"Optimizer states: {mem.optimizer_states_gb:.2f} GB")
print(f"Activations:      {mem.activations_gb:.2f} GB")
print(f"CUDA overhead:    {mem.overhead_gb:.2f} GB")
print(f"TOTAL:            {mem.total_gb:.2f} GB")
```

**FlashAttention-2** avoids materializing the full N×N attention matrix, cutting activation memory by ~50%:

```
Without FlashAttention: 9.09 GB
With FlashAttention:    6.79 GB  ← saved 2.30 GB (25%)
```

**Supported methods:** Full Fine-Tuning, LoRA, QLoRA (4-bit / 8-bit)
**Supported optimizers:** AdamW, Adam, SGD, 8-bit Adam (bitsandbytes), Adafactor

### ⏱️ Training Time Estimation

FLOPs-based wall-clock time estimates with multi-GPU scaling:

```python
# Single GPU
time = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3)

# Compare all compatible GPUs
for t in est.estimate_time_all_gpus(dataset_size=50000, epochs=3):
    print(f"{t.gpu_name:<18} {t.total_hours:>6.1f}h")
```

### 💰 Cloud Cost Comparison

Compare across **8 cloud providers** including spot pricing:

```python
costs = est.full_comparison(dataset_size=50000, epochs=3)
print(f"🏆 Cheapest: {costs.cheapest}")
print(f"💡 Best value: {costs.best_value}")
```

**Providers:** AWS, Google Cloud, Microsoft Azure, Lambda Labs, RunPod, Vast.ai, Together AI, Modal

---

## 🎯 Budget Optimizer

Reverse the logic — tell ftune your constraints and it finds the optimal configuration:

```python
from ftune import BudgetOptimizer

recs = BudgetOptimizer.optimize(
    model="meta-llama/Llama-3.1-8B",
    budget=50.0,              # max $50
    gpu="RTX-3090-24GB",      # hardware constraint
    dataset_size=10000,
    epochs=1,
    priority="cost",          # "cost", "speed", or "quality"
)

print(BudgetOptimizer.format_recommendations(recs))
```

```
╭────────────────────────────────────────────────╮
│    🎯 ftune Budget Optimizer — Recommendations  │
╰────────────────────────────────────────────────╯

  #1: LORA none, rank=8
      GPU: RTX-3090-24GB (Vast.ai) | 1 GPU(s)
      Batch: 1 × 1 accum | Optimizer: adamw
      Memory: 17.8 GB | Time: 8.3h | Cost: $2.07
      💡 FlashAttention-2 enabled, Gradient checkpointing ON

  #2: LORA none, rank=16
      GPU: RTX-3090-24GB (Vast.ai) | 1 GPU(s)
      Batch: 1 × 1 accum | Optimizer: adamw
      Memory: 17.9 GB | Time: 8.3h | Cost: $2.07
      💡 FlashAttention-2 enabled, Gradient checkpointing ON
```

The optimizer searches across methods, LoRA ranks, batch sizes, optimizers, and FlashAttention to find configurations that fit your budget and hardware.

---

## 🔀 Multi-GPU & Sharding

ftune supports DeepSpeed ZeRO Stages 1/2/3 and PyTorch FSDP for multi-GPU memory estimation:

```python
# Single GPU — doesn't fit
est = Estimator(model="meta-llama/Llama-3.1-8B", method="full", batch_size=1)
print(est.estimate_memory().total_gb)  # 104.4 GB ❌

# ZeRO-3 on 4 GPUs — fits!
est = Estimator(
    model="meta-llama/Llama-3.1-8B",
    method="full",
    batch_size=1,
    sharding="zero_3",
    num_gpus=4,
)
print(est.estimate_memory().total_gb)  # 27.8 GB per GPU ✅
```

**How sharding reduces per-GPU memory:**

| Strategy | What's sharded | 8B Full FT (per GPU) |
|---|---|---|
| None (single GPU) | Nothing | 104.4 GB |
| ZeRO Stage 1 | Optimizer states | 52.8 GB |
| ZeRO Stage 2 | + Gradients | 39.9 GB |
| ZeRO Stage 3 / FSDP | + Model weights | 27.8 GB |

This means ftune can now tell you: *"This 70B model won't fit on one A100, but it will fit on 4×A100s using ZeRO-3 with 17.4 GB per GPU utilization."*

```python
est = Estimator(
    model="meta-llama/Llama-3.1-70B",
    method="qlora",
    quantization="4bit",
    sharding="zero_3",
    num_gpus=4,
)
mem = est.estimate_memory()
print(f"70B QLoRA ZeRO-3: {mem.total_gb:.1f} GB per GPU")  # 17.4 GB ✅
```

**Supported strategies:** `none`, `zero_1`, `zero_2`, `zero_3`, `fsdp`, `fsdp_shard_grad`

---

## 🔧 Calibration

Generic MFU constants can be off by 2-10x depending on your hardware, drivers, and framework. Calibration fixes this.

**Run a quick 10-step benchmark on your GPU, then feed the results to ftune:**

```python
from ftune import Estimator, Calibrator

est = Estimator(model="meta-llama/Llama-3.1-8B", method="qlora", quantization="4bit")
mem = est.estimate_memory()
time = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3)

# After running 10 real training steps, you measured:
cal = Calibrator.from_benchmark(
    estimated_memory_gb=mem.total_gb,     # ftune's estimate
    actual_memory_gb=11.2,                 # nvidia-smi peak
    estimated_time_hours=time.total_hours, # ftune's estimate
    actual_time_hours=5.0,                 # extrapolated from benchmark
    gpu_name="A100-80GB",
)

# Now all future estimates are hardware-calibrated
adjusted_time = cal.adjust_time(time.total_hours)
adjusted_memory = cal.adjust_memory(mem.total_gb)
print(f"Calibrated time: {adjusted_time:.1f}h (was {time.total_hours:.1f}h)")
print(f"Calibrated memory: {adjusted_memory:.1f} GB (was {mem.total_gb:.1f} GB)")

# Save calibration for reuse
Calibrator.save(cal.result, "~/.ftune/my_a100_calibration.json")

# Load it later
from ftune.core.models import CalibrationResult
saved = Calibrator.load("~/.ftune/my_a100_calibration.json")
```

**You can also use the measured MFU directly:**

```python
# Calibration found your actual MFU is 0.52
time = est.estimate_time(gpu="A100-80GB", dataset_size=50000, epochs=3, mfu_override=0.52)
```

---

## 📊 Validation

Compare ftune estimates against actual training metrics from real runs:

```python
from ftune import Estimator
from ftune.validation import Validator, ActualMetrics

est = Estimator(model="meta-llama/Llama-3.1-8B", method="qlora", quantization="4bit")

actual = ActualMetrics(
    peak_memory_gb=11.2,
    training_time_hours=4.5,
    total_cost=8.50,
    gpu_name="A100-80GB",
    dataset_size=50000,
    epochs=3,
)

result = Validator.compare(est, actual)
print(Validator.format_report(result))
```

**Load metrics from multiple sources:**

```python
actual = Validator.from_json("training_metrics.json")
actual = Validator.from_wandb("username/project/run_id")  # pip install ftune[wandb]
metrics_list = Validator.from_csv("all_runs.csv")          # batch validation
```

---

## ⌨️ CLI

```bash
pip install ftune[cli]
```

```bash
# Full estimate
ftune estimate --model meta-llama/Llama-3.1-8B --method qlora --quantization 4bit

# With all options
ftune estimate \
  --model meta-llama/Llama-3.1-8B \
  --method qlora \
  --quantization 4bit \
  --lora-rank 16 \
  --batch-size 4 \
  --seq-length 2048 \
  --dataset-size 50000 \
  --epochs 3 \
  --output json

# List models and GPUs
ftune models
ftune gpus

# Check pricing
ftune pricing --gpu A100-80GB --hours 10

# Validate against actual metrics
ftune validate --model meta-llama/Llama-3.1-8B --method qlora --metrics metrics.json
```

---

## 🧮 How It Works

### Memory Formula

| Component | Full Fine-Tune | LoRA | QLoRA (4-bit) |
|---|---|---|---|
| Model Weights | `params × dtype_bytes` | `params × dtype_bytes` | `params × 0.5` + quant overhead |
| Trainable Params | (same as weights) | `modules × 2 × hidden × rank` | Same as LoRA |
| Gradients | `params × 2B` | `lora_params × 2B` | `lora_params × 2B` |
| Optimizer (AdamW) | `params × 8B` | `lora_params × 8B` | `lora_params × 8B` |
| Optimizer (8-bit Adam) | `params × 2B` | `lora_params × 2B` | `lora_params × 2B` |
| Activations | `batch × seq × hidden × layers × factor` | Same | Same |
| FlashAttention-2 | Activations × 0.5 | Activations × 0.5 | Activations × 0.5 |
| Overhead | ~15% buffer | ~15% buffer | ~15% buffer |
| ZeRO-3 / FSDP | Weights, grads, optimizer ÷ N GPUs | Same | Same |

### Training Time Formula

```
FLOPs per token ≈ 6 × num_parameters
Total FLOPs = flops_per_token × dataset_tokens × epochs
Time = Total FLOPs / (GPU TFLOPS × MFU × num_gpus × scaling_efficiency)
```

MFU defaults to 0.30-0.35 (conservative). Use calibration mode for hardware-specific values.

---

## 📋 Supported Models, GPUs & Providers

### Built-in Models (15+)

| Model | Parameters | Default dtype |
|---|---|---|
| meta-llama/Llama-3.1-8B / 70B / 405B | 8B / 70B / 405B | bf16 |
| mistralai/Mistral-7B-v0.3 | 7B | bf16 |
| mistralai/Mixtral-8x7B-v0.1 | 47B (MoE) | bf16 |
| google/gemma-2-9b / 27b | 9B / 27B | bf16 |
| Qwen/Qwen2.5-7B / 72B | 7B / 72B | bf16 |
| microsoft/phi-3-mini / medium | 3.8B / 14B | bf16 |
| deepseek-ai/DeepSeek-V2-Lite | 16B | bf16 |
| + Yi, Falcon, StableLM | various | bf16 |

> **Plus ANY model on HuggingFace Hub** via auto-detect.

### GPUs (11)

| GPU | VRAM | FP16 TFLOPS |
|---|---|---|
| NVIDIA H100 | 80 GB | 989 |
| NVIDIA A100 | 40 / 80 GB | 312 |
| NVIDIA A10G | 24 GB | 125 |
| NVIDIA L4 | 24 GB | 121 |
| RTX 4090 / 4080 | 24 / 16 GB | 165 / 97 |
| RTX 3090 | 24 GB | 71 |
| Tesla T4 / V100 | 16 / 16-32 GB | 65 / 125 |

### Cloud Providers (8)

| Provider | GPUs Available | Spot Pricing |
|---|---|---|
| AWS | H100, A100, T4, L4 | ✅ |
| Google Cloud | H100, A100, T4, L4, V100 | ✅ |
| Microsoft Azure | H100, A100, T4, V100 | ✅ |
| Lambda Labs | H100, A100, RTX 4090 | — |
| RunPod | H100, A100, RTX 4090/3090, L4 | ✅ |
| Vast.ai | H100, A100, RTX 4090/3090 | — |
| Together AI | H100, A100 | — |
| Modal | H100, A100, L4, T4 | — |

---

## 📦 Installation

```bash
pip install ftune            # Core library (zero ML dependencies)
pip install ftune[cli]       # + CLI with Rich terminal output
pip install ftune[web]       # + Streamlit web UI
pip install ftune[wandb]     # + Weights & Biases validation
pip install ftune[all]       # Everything
```

---

## 🗺️ Roadmap

- [x] Memory estimation (full, LoRA, QLoRA)
- [x] Training time estimation (FLOPs-based, multi-GPU)
- [x] Cloud cost comparison (8 providers, spot pricing)
- [x] HuggingFace Hub auto-detect
- [x] FlashAttention-2 memory optimization
- [x] ZeRO Stages 1/2/3 + FSDP sharding
- [x] Hardware calibration mode
- [x] Budget optimizer ("I have $50, what's optimal?")
- [x] Validation mode (manual, JSON, W&B, CSV)
- [x] Streamlit web UI
- [x] CLI with Rich output
- [x] GitHub Actions CI/CD
- [ ] Streamlit Cloud deployment (public hosted version)
- [ ] PyPI publish
- [ ] Community validation dataset (crowdsourced accuracy data)
- [ ] Active pricing API (real-time provider rates)
- [ ] Exportable PDF reports

---

## 🔑 Design Principles

- **Zero ML dependencies** — Pure Python calculator. No PyTorch, no TensorFlow, no GPU required.
- **Works with any model** — HuggingFace Hub integration for instant support of thousands of models.
- **Hardware-aware** — Calibration mode closes the gap between theory and your specific setup.
- **Enterprise-ready** — ZeRO/FSDP sharding makes ftune relevant for serious multi-GPU training.
- **Validates itself** — Compare estimates against actual runs. No black box.
- **Fast** — Every estimate returns in under 1 second.

---

## 🤝 Contributing

The most valuable contribution is **validation data**. Run ftune against your actual training runs and share the results:

```python
from ftune import Estimator, Calibrator

est = Estimator(model="your-model", method="qlora", quantization="4bit", ...)
cal = Calibrator.from_benchmark(
    estimated_memory_gb=est.estimate_memory().total_gb,
    actual_memory_gb=...,       # from nvidia-smi
    estimated_time_hours=est.estimate_time(...).total_hours,
    actual_time_hours=...,       # wall-clock
    gpu_name="...",
)
print(cal.format_report())
# Share this in an issue or PR!
```

Other ways to help: update pricing data, add models/GPUs, fix bugs, improve the web UI.

```bash
git clone https://github.com/ritikmahy5/ftune.git
cd ftune
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

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
