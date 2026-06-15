#!/bin/bash
#SBATCH --job-name=dominant_replay
#SBATCH --partition=plgrid
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

module load gcc/11.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

uv run fraud-detection --data-dir data --cpu --model dominant --strategy replay --mode continual --replay-buffer-size 20000 --epochs 30 --batch-size 8192
