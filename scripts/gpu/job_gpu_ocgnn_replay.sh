#!/usr/bin/env bash
#SBATCH --job-name=gpu_ocgnn_replay
#SBATCH --account=plgmpr26-gpu
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

module load cuda/12.4.0
module load gcc/11.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

uv run fraud-detection --data-dir data --model ocgnn --strategy replay --mode continual --replay-buffer-size 20000 --epochs 100 --batch-size 32768 --num-neigh 10
