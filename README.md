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

- **train** → `data/<sampler>/checkpoints/model_d{d}_K{K}_s{seed}.pt` (one per repeat seed) and
  `data/<sampler>/manifest.json`. Idempotent: existing checkpoints are reused unless
  `train.force_retrain=true`.
- **evaluate** → `output/<run_id>/<sampler>/T<T_true>/results.json`, `results.csv` (the
  **mean ± std over seeds**, one row per `(d, K, budget, model)` with `<metric>` = mean and
  `<metric>_std`) and `results_per_seed.csv` (the raw per-seed rows). For each `(d,K)`: the
  ground-truth hallucination rate and, for every anchor count in `sweep.anchors`, every
  predictor's full/mode/hall accuracy and F1.
- **visualize** → heatmaps and tables under that run's `figures/<sampler>/T<T_true>/`:
  `hallucination_rate.png` and `responsibility.png` (both anchor-free), plus one
  `anchors_<n>/` per budget holding `atlas.png`, `table.tex` and `table.png`. Reads only
  `results.json`, so you can restyle without recomputing.

## Improved recipe (the accuracy fixes)

The defaults now follow the tuned recipe. Four changes lift the numbers over the earlier run:

1. **Trainer (biggest gain).** `core.train_learned` / the flow trainer now cosine-decay the lr
   (`train.lr` → `train.lr_min`), clip the gradient norm (`train.grad_clip`), and keep an EMA of
   the weights (`train.ema_decay`, warmed up over `train.ema_warmup`) that replaces the raw weights
   at the end. `train.base_steps` is 30 000 and `train.hall_target` is 0.015. This is what fixes the
   broken high-`d` DDIM ground truth (probe hall was reaching 0.2–0.9). Disable any piece by setting
   it `null`.
2. **Second-order exact-field backtrack.** `core.backtrack_true`/`forward_true` take a Heun
   predictor–corrector step (`process.true_order=heun` for DDIM, `process.true_solver=heun` for
   flow). Data → seed → data now round-trips to the label exactly (verified: label-match 1.0000).
3. **Prior calibration.** Every parametric predictor also emits a `<name>_cal` row: its
   hallucination logit is shifted so it calls exactly the exact-score hallucination fraction
   (5 000 calibration seeds).
4. **Stable, cached ground truth.** `eval.n_eval` is 200 000 (stable ~1–2 % hallucination metrics),
   and the ground-truth labels are cached at train time under `data/<sampler>/gt_cache/`, so a whole
   `T_true` sweep runs the N-seed forward pass at most once per `(d, K)`. Delete `gt_cache/` to force
   a recompute.

### Why `T_train` is 500 (was 150)

The learned sampler's hallucination rate at `T_train=150` exploded with dimension (d=32: 10-44 %,
d=64: 57-83 %, d=128: 99 %) -- and so does the **exact** score pushed through the same 150-step
Euler DDIM (15 / 52 / 93 / 100 % at d = 32 / 64 / 128 / 256). It is discretisation error, not
training error: a mode's 99 % ball is a shell of relative width ~1.2/sqrt(d), so a few percent of
systematic variance error from the coarse integration ejects every sample in high d (and the
linear beta schedule only reaches abar_T = 0.22 at T=150). With the exact score, T=300 still fails
at d >= 128, while T=500 gives <= 0.6 % and T=1000 <= 0.8 % at every d up to 256. Training cost
does not depend on T; only the sampling passes do.

### Repeat seeds: every number is a mean ± std

`n_seeds` (default 3) repeats the whole experiment for seeds `seed, seed+100, seed+200, ...`
(`core.seed_list`). Each repeat re-draws the mode placement on the sphere, the model init and the
evaluation seeds, so the std is over independent geometries, not just over training noise. Every
metric in the console, `results.csv`, the LaTeX/PNG tables and the heatmaps is reported as
mean ± std over those seeds; the best anchor budget per cell is chosen by mean full accuracy.

### Fast, full-GPU sweep

The score nets are small MLPs at batch 512, so an eager training loop is bound by kernel-launch
overhead (~6 ms of CPU per ~0.5 ms of GPU work: ~20 % GPU utilisation). `core.run_optimizers`
therefore captures one whole optimiser step -- for **all seeds of a cell at once, one CUDA stream
per seed** -- into a single CUDA graph and replays it (`train.cuda_graph`, `train.graph_streams`;
same maths, ~15x the eager throughput per GPU). `train.cuda_graph=false` recovers the eager loop.

`scripts/main.sh` fills every GPU: phase 1 launches **one training job per `(process, d, K)` cell**
(all `NSEEDS` seeds of the cell trained together; distinct checkpoint files, so no races), phase 2
one eval+viz job per `(process, T)`. Failed jobs are retried once (`RETRIES`). Keep
`JOBS_PER_GPU=1` (the default): a job already trains its seeds concurrently, and several such
processes on one GPU intermittently crash inside the driver, so `JOBS_PER_GPU>1` automatically
falls back to single-stream graphs.

```bash
./scripts/main.sh                                  # full grid, 3 seeds, 4 GPUs
NSEEDS=5 DIMS="2 4 8 16" ./scripts/main.sh         # 5 repeats on a narrower dimension sweep
DEVICE=cpu NGPU=2 DIMS="2" KS="2" TS="200" ANCHORS="[2000]" ./scripts/main.sh   # tiny CPU smoke
```

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
