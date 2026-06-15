#!/bin/bash
#SBATCH --job-name=ocgnn_replay
#SBATCH --partition=plgrid
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

module load gcc/11.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

uv run fraud-detection --data-dir data --cpu --model ocgnn --strategy replay --mode continual --replay-buffer-size 20000 --epochs 30 --batch-size 8192
