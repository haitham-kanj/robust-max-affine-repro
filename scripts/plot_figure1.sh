#!/usr/bin/env bash
set -euo pipefail

DATA="results/figure1/sweeper_total_l1_nospectral_k4_d50-200-25_n500-3000-250_alpha0p2_M50p0_Mrand50_scalefirst_ref5_finaliter50_threshold0p001_m20_seed1.npz"
${PYTHON:-python3} Codes/Algorithms/view_sweeper.py \
  --data "$DATA" \
  --output figures/phase_transition.pdf \
  --statistic median \
  --cap 1
