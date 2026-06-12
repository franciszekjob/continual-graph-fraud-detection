#!/bin/bash
#SBATCH --job-name=cola_naive
#SBATCH --partition=plgrid
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

module load gcc/11.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

uv run fraud-detection --data-dir data --cpu --model cola --strategy naive --mode both --epochs 10 --batch-size 4096
