#!/bin/bash
#SBATCH --job-name=ram_figure1_gpu
#SBATCH --partition=gpu_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --output=logs/figure1_gpu_%j.out
#SBATCH --error=logs/figure1_gpu_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs .cache/cupy
source pygpu/bin/activate
bash scripts/run_figure1_gpu.sh
