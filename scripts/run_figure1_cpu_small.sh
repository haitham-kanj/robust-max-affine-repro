#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/sweeper figures
${PYTHON:-python3} -u Codes/Algorithms/sweeper.py \
  --nstart 100 \
  --nend 200 \
  --njump 100 \
  --dstart 10 \
  --dend 20 \
  --djump 10 \
  --m 2 \
  --k 4 \
  --alpha 0.2 \
  --M 50 \
  --Mrand 5 \
  --refinement-steps 2 \
  --max-iterations 20 \
  --threshold 1e-3 \
  --seed 1
