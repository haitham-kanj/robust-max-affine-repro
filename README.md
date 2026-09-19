# Robust Max-Affine Regression Reproducibility Code

This repository contains the minimal code and cached outputs needed to reproduce Figure 1 and Figure 2 from the paper.

The repository includes:

- `Codes/Algorithms/`: CPU RAM, spectral baseline, synthetic sweeper, EPA benchmark scripts, and plotting code.
- `Codes/AlgorithmsGPU/`: GPU synthetic sweeper used for Figure 1.
- `data/epa/`: frozen EPA CAMPD subset used for the real-data benchmark.
- `results/figure1/`: cached synthetic sweep output used to regenerate Figure 1.
- `results/figure2/`: cached EPA benchmark outputs used to report Figure 2 values.
- `figures/`: generated figure files.
- `instructions.md`: exact environment setup and run commands.

For a quick check after installing the CPU environment, run

```bash
bash scripts/plot_figure1.sh
bash scripts/print_figure2.sh
```

The full synthetic sweep for Figure 1 is GPU/Slurm oriented and can take substantial time. The cached `.npz` output is included so the figure can be regenerated without rerunning the sweep.
