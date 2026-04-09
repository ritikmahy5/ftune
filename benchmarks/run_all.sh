#!/bin/bash
# Submit all benchmark jobs to Slurm
# Run from the benchmarks/ directory

mkdir -p logs results

echo "Submitting benchmark jobs..."

for job in jobs/*.sh; do
    echo "  Submitting $job"
    sbatch "$job"
done

echo ""
echo "Jobs submitted. Monitor with: squeue -u $USER"
echo "Results will appear in benchmarks/results/"
echo ""
echo "After all jobs complete, copy results back and run:"
echo "  python summarize.py results/"
