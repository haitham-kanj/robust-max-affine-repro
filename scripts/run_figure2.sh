#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/real_data
DATA="data/epa/campd_california_q1_2019_600.json"

${PYTHON:-python3} -u Codes/Algorithms/epa_real_data_k_sweep.py \
  --data-path "$DATA" \
  --kstart 4 \
  --kend 5 \
  --Mrand 20 \
  --refinement-steps 5 \
  --splits 10 \
  --seed 1

${PYTHON:-python3} -u Codes/Algorithms/epa_paper_methods.py \
  --data-path "$DATA" \
  --splits 10 \
  --seed 1
