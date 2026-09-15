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
│   ├── visualize.py   # STAGE 3: results.json -> figures                -> visualization/
│   └── main.py        # Hydra entry point dispatching the stages
├── conf/
│   └── config.yaml    # the single source of truth for the sweep + all knobs
├── data/              # checkpoints/  and  manifest.json   (generated)
├── output/            # results.json, results.csv, hydra run logs (generated)
└── visualization/     # PNG figures (generated)
```

## Install

```bash
pip install -r requirements.txt
```

## Run

Set the repo root once (so the `paths.*` interpolations resolve), then call `main.py`:

```bash
export ATLAS_ROOT=$(pwd)          # run this from the diffusion_atlas/ folder
python code/main.py               # full pipeline: train -> evaluate -> visualize
```

### Configure the sweep

Everything is overridable on the command line (Hydra):

```bash
# custom dimension / mode / anchor sweep
python code/main.py sweep.d=[2,8,32] sweep.K=[8] sweep.anchors=[5000,50000,200000]

# change the true-score backtrack resolution and the vote bandwidth
python code/main.py sweep.T_true=200 eval.h_frac=0.2

# run only some stages (e.g. just re-plot from an existing results.json)
python code/main.py stages=[visualize]
python code/main.py stages=[evaluate,visualize]

# force retrain, run on CPU
python code/main.py train.force_retrain=true device=cpu
```

Or just edit `conf/config.yaml`.

## What each stage writes

- **train** → `data/checkpoints/model_d{d}_K{K}.pt` and `data/manifest.json`.
  Idempotent: existing checkpoints are reused unless `train.force_retrain=true`.
- **evaluate** → `output/results.json` (full, nested) and `output/results.csv` (flat).
  For each `(d,K)`: the ground-truth hallucination rate, the analytic-responsibility
  accuracy, and, for every anchor count in `sweep.anchors`, the atlas full/class accuracy
  and the bandwidth used.
- **visualize** → `visualization/accuracy_vs_dimension.png`,
  `accuracy_vs_anchors.png`, `hallucination_vs_dim.png`. Reads only `results.json`,
  so you can restyle without recomputing.

## Notes

- `R99` is derived from the chi-square quantile `mass_q` and carries the `sqrt(d)`
  scaling, so the Gaussian-vote bandwidth `h = h_frac * R99` adapts with dimension.
- The atlas is built with the **true score** (backtracked disk + shell); the ground
  truth uses the **learned** DDIM sampler. Comparing them is the point.
- The analytic-responsibility predictor uses **no anchors** and is the dimension-robust
  baseline; expect its class accuracy to hold (or improve) as `d` grows while the raw
  atlas vote degrades unless the anchor budget grows.
