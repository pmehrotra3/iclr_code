# Diffusion Atlas — configurable sweep pipeline

A Hydra-configurable pipeline that (1) trains learned diffusion models over a `(d, K)`
sweep and saves checkpoints, (2) compares a true-score backtracked atlas (all-anchor
Gaussian vote) and an analytic-responsibility predictor against the learned-model
ground-truth fate over an anchor sweep, and (3) generates all figures from the saved
results.

## Layout

```
diffusion_atlas/
├── code/
│   ├── core.py        # shared numerics: models, schedule, GMM, samplers, atlas, vote
│   ├── train.py       # STAGE 1: train + save one checkpoint per (d,K)  -> data/
│   ├── evaluate.py    # STAGE 2: atlas + responsibility vs true score   -> output/
│   ├── visualize.py   # STAGE 3: results.json -> figures (beside the results)
│   └── main.py        # Hydra entry point dispatching the stages
├── conf/
│   ├── config.yaml    # shared knobs + the defaults list (which process, which sweep)
│   ├── process/       # per-process knobs: ddim.yaml, flow.yaml
│   └── sweep/         # the (d, K, anchors) grid: default.yaml
├── data/              # checkpoints/  and  manifest.json   (generated)
└── output/            # per run: <sampler>/<run_id>/{results.json, results.csv, figures/}
```

## Install

```bash
pip install -r requirements.txt
```

## Samplers (DDIM vs Flow Matching)

The pipeline supports two generative processes, selected by the `process` group:

- `process=ddim` — variance-preserving diffusion, deterministic DDIM (default).
- `process=flow` — Flow Matching with the Optimal-Transport path (Lipman et al. 2022,
  Eq. 20–23): `psi_t(x0) = (1-(1-sigma_min)t) x0 + t x1`, velocity target
  `x1 - (1-sigma_min) x0`, ODE-integrated for sampling. Solver is configurable
  (`process.solver = euler | midpoint | rk4`).

Runs are **fully separate**: pick one process per run. Its checkpoints and results live
under `data/<sampler>/` and `output/<sampler>/<run_id>/` (figures beside them, in
`figures/`), so DDIM and Flow runs never overwrite each other.

```bash
export ATLAS_ROOT=$(pwd)

python code/main.py process=ddim                          # diffusion run
python code/main.py process=flow                          # flow-matching (OT) run
python code/main.py process=flow process.solver=midpoint  # OT with midpoint solver
python code/main.py process=flow process.solver=rk4 process.T_train=50
```

## Run

Set the repo root once (so the `paths.*` interpolations resolve), then call `main.py`:

```bash
export ATLAS_ROOT=$(pwd)          # run this from the diffusion_atlas/ folder
python code/main.py               # full pipeline (default process=ddim)
```

### Configure the sweep

Everything is overridable on the command line (Hydra):

```bash
# custom dimension / mode / anchor sweep
python code/main.py sweep.d=[2,8,32] sweep.K=[8] sweep.anchors=[5000,50000,200000]

# change the true-score backtrack resolution and the vote bandwidth
python code/main.py process.T_true=200 eval.h_frac=0.2

# re-plot a specific past run (otherwise visualize falls back to the newest one)
python code/main.py stages=[visualize] run_id=2026-09-18_13-04-22

# run only some stages (e.g. just re-plot from an existing results.json)
python code/main.py stages=[visualize]
python code/main.py stages=[evaluate,visualize]

# force retrain, run on CPU
python code/main.py train.force_retrain=true device=cpu
```

Or just edit `conf/config.yaml`.

## What each stage writes

- **train** → `data/<sampler>/checkpoints/model_d{d}_K{K}.pt` and `data/<sampler>/manifest.json`.
  Idempotent: existing checkpoints are reused unless `train.force_retrain=true`.
- **evaluate** → `output/<sampler>/<run_id>/results.json` (full, nested) and `results.csv`
  (flat). For each `(d,K)`: the ground-truth hallucination rate, the analytic-responsibility
  accuracy, and, for every anchor count in `sweep.anchors`, the atlas full/mode/hall accuracy
  and the bandwidth used.
- **visualize** → heatmaps and tables under that run's `figures/<sampler>/T<T_true>/`:
  `hallucination_rate.png` and `responsibility.png` (both anchor-free), plus one
  `anchors_<n>/` per budget holding `atlas.png`, `table.tex` and `table.png`. Reads only
  `results.json`, so you can restyle without recomputing.

## Notes

- `R99` is derived from the chi-square quantile `mass_q` and carries the `sqrt(d)`
  scaling, so the Gaussian-vote bandwidth `h = h_frac * R99` adapts with dimension.
- The atlas is built with the **true score** (backtracked disk + shell); the ground
  truth uses the **learned** DDIM sampler. Comparing them is the point.
- The analytic-responsibility predictor uses **no anchors** and is the dimension-robust
  baseline; expect its class accuracy to hold (or improve) as `d` grows while the raw
  atlas vote degrades unless the anchor budget grows.

## Figures a run produces

Outputs per run under `output/<sampler>/<run_id>/figures/<sampler>/T<T_true>/`:
- `hallucination_rate.png`, `responsibility.png` — the anchor-free panels over the (d,K) grid.
- `anchors_<n>/atlas.png` — the atlas full/mode/hall heatmaps at that anchor budget.
- `anchors_<n>/table.tex` — a booktabs LaTeX table (one row per (K,d): responsibility vs
  atlas at that budget; any mislabel counts as wrong). Requires `\usepackage{booktabs}`.
  `table.png` is the same table as an image. Each anchor budget gets its own directory — no
  selection over the sweep.

Run each process on its own: `python code/main.py process=ddim`, then
`python code/main.py process=flow`. Each writes to its own `output/<sampler>/<run_id>/`,
and `process.T_true=<val>` sweeps the backtrack resolution.
