#!/bin/bash
#SBATCH --job-name=ftune-llama8b-lora-a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/llama8b_lora_a100_%j.out

module load anaconda3
module load cuda/12.1
conda activate ftune-bench

cd $SLURM_SUBMIT_DIR/..

python benchmark.py \
    --model meta-llama/Llama-3.1-8B \
    --method lora \
    --quantization none \
    --lora-rank 16 \
    --lora-target attention \
    --batch-size 4 \
    --seq-length 2048 \
    --num-steps 20 \
    --gpu A100-80GB \
    --output results/llama8b_lora_a100.json
