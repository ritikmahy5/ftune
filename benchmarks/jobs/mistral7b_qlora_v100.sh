#!/bin/bash
#SBATCH --job-name=ftune-mistral7b-v100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/mistral7b_qlora_v100_%j.out

module load anaconda3
module load cuda/12.1
conda activate ftune-bench

cd $SLURM_SUBMIT_DIR/..

python benchmark.py \
    --model mistralai/Mistral-7B-v0.3 \
    --method qlora \
    --quantization 4bit \
    --lora-rank 16 \
    --lora-target attention \
    --batch-size 1 \
    --seq-length 512 \
    --num-steps 20 \
    --gpu V100-16GB \
    --output results/mistral7b_qlora_v100.json
