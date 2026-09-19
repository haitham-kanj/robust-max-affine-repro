#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs .cache/cupy results/sweeper figures
export CUPY_CACHE_DIR="$PWD/.cache/cupy"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

${PYTHON:-python3} -u Codes/AlgorithmsGPU/sweeper_gpu.py \
  --nstart 500 \
  --nend 3000 \
  --njump 250 \
  --dstart 50 \
  --dend 200 \
  --djump 25 \
  --m 20 \
  --k 4 \
  --alpha 0.2 \
  --M 50 \
  --Mrand 50 \
  --refinement-steps 5 \
  --max-iterations 50 \
  --threshold 1e-3 \
  --gpus 0,1,2,3 \
  --seed 1
