# Contributing to ftune

Thanks for your interest in contributing! Here's how you can help.

## 🏆 Highest-Impact Contributions

### 1. Validate Estimates (Most Wanted!)
Run ftune against your actual training runs and share the results. This helps everyone calibrate expectations.

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
```

Open an issue or PR with your validation report!

### 2. Update Pricing Data
Cloud GPU pricing changes frequently. Help keep `src/ftune/data/pricing.yaml` current.

### 3. Add Models
Add new model specs to `src/ftune/data/models.yaml`. Include: parameters, hidden_size, num_layers, num_attention_heads, num_kv_heads, intermediate_size, vocab_size, max_seq_length.

### 4. Add GPUs
Add new GPU specs to `src/ftune/data/gpus.yaml`. Include: vram_gb, fp16_tflops, bf16_tflops, fp32_tflops, memory_bandwidth_gbps.

## Development Setup

```bash
git clone https://github.com/ritikmahy5/ftune.git
cd ftune

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
PYTHONPATH=src pytest tests/ -v

# Run the CLI (needs extras)
pip install typer rich
PYTHONPATH=src python -m ftune estimate --model meta-llama/Llama-3.1-8B --method qlora

# Run the web UI
pip install streamlit pandas altair
PYTHONPATH=src streamlit run src/ftune/app.py
```

## Code Style

- Type hints on all functions
- Docstrings on all public methods
- Keep zero ML dependencies in core (no torch/transformers)
- Run `ruff check src/` before submitting

## Pull Request Process

1. Fork the repo and create a feature branch
2. Add tests for any new functionality
3. Ensure all tests pass: `PYTHONPATH=src pytest tests/ -v`
4. Submit a PR with a clear description

## Reporting Issues

- For inaccurate estimates: include your model, config, actual metrics, and ftune output
- For bugs: include Python version, OS, and full traceback
- For feature requests: describe the use case and expected behavior
