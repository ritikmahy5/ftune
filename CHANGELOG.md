# Changelog

## [0.4.0] - 2026-04-09

### Added
- H200-141GB GPU support and benchmark validation
- 6 new models: Llama 3.3 70B, Qwen3 8B/32B, DeepSeek V3, Gemma 3 12B, Phi-4
- MoE-aware LoRA parameter calculation (per-expert FFN layers)
- Validation results table in README (5 benchmarks, 3 GPU architectures)

### Fixed
- GQA-aware LoRA parameter calculation using actual projection dimensions
- Sharding-aware multi-GPU time scaling (per-strategy: DDP, ZeRO-1/2/3, FSDP)
- V100/T4 bf16_tflops corrected to fp32 speed (no native bf16 tensor cores)
- Activation factor calibrated against real benchmarks (2.0 -> 4.0)
- CUDA overhead fraction adjusted (0.15 -> 0.20)
- QLoRA dequantization workspace overhead added
- All mypy type errors resolved
- All ruff lint warnings resolved

## [0.3.0] - 2026-04-09

### Added
- Published to PyPI as `ftuneai`
- `ftune pricing-update` CLI command for updating GPU prices
- `ftune pricing --stale N` for staleness reporting
- Per-provider `last_updated` timestamps in pricing database
- Source citations for all estimation constants
- Limitations & Accuracy section in README

### Changed
- Renamed Python package from `ftune` to `ftuneai` for consistency with PyPI name

## [0.2.0] - 2026-02-10

### Added
- Initial release
- Memory estimation (full fine-tuning, LoRA, QLoRA)
- Training time estimation (FLOPs-based, multi-GPU)
- Cloud cost comparison (8 providers, spot pricing)
- HuggingFace Hub auto-detect
- FlashAttention-2 memory optimization
- ZeRO Stages 1/2/3 + FSDP sharding
- Hardware calibration mode
- Budget optimizer
- Validation mode (manual, JSON, W&B, CSV)
- Streamlit web UI
- CLI with Rich output
- GitHub Actions CI/CD
