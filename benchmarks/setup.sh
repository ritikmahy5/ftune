#!/bin/bash
# Setup conda environment on Northeastern Discovery cluster
# Run this once before submitting benchmark jobs

module load anaconda3
module load cuda/12.1

conda create -n ftune-bench python=3.10 -y
conda activate ftune-bench

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes datasets accelerate ftuneai

echo "Setup complete. Run: conda activate ftune-bench"
