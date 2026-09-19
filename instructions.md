# Reproduction Instructions

Run all commands from the repository root.

## 1. CPU environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Regenerate Figure 1 from cached results

```bash
bash scripts/plot_figure1.sh
```

This reads the cached synthetic sweep file in `results/figure1/` and writes `figures/phase_transition.pdf`.

## 3. Print Figure 2 benchmark numbers from cached results

```bash
bash scripts/print_figure2.sh
```

This prints the mean and standard deviation across the 10 fixed EPA train/test splits.

## 4. Recreate Figure 1 sweep on GPU

Create the GPU environment on a CUDA 12 system:

```bash
python3 -m venv pygpu
source pygpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
```

Run interactively on a machine with four visible GPUs:

```bash
bash scripts/run_figure1_gpu.sh
```

Or submit the Slurm script:

```bash
sbatch slurm/submit_figure1_gpu.sbatch
```

The GPU sweep writes a new `.npz` file under `results/sweeper/`. Regenerate the plot by passing that file to `Codes/Algorithms/view_sweeper.py` or by copying it into `results/figure1/` and rerunning `scripts/plot_figure1.sh`.

## 5. Small CPU smoke test

The full Figure 1 grid is intended for the GPU script. To check that the CPU code runs on a small grid:

```bash
bash scripts/run_figure1_cpu_small.sh
```

## 6. Recreate Figure 2 EPA results

The repository includes the frozen EPA data file used for the paper. To rerun the RAM k-sweep and comparison baselines:

```bash
bash scripts/run_figure2.sh
```

The scripts write outputs under `results/real_data/`. These runs can take time because the baseline optimization problems are solved on repeated train/test splits.
