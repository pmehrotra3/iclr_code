# Seed-fate atlas

Which initial noise seeds make a diffusion (or flow-matching) model **hallucinate**?

Data is a Gaussian mixture of K modes on a sphere in d dimensions. A seed x ~ N(0, I) has a
**fate**: the mode whose 99%-mass ball (radius R99) its sample lands in, or *hallucination* if it
lands in none. Because the mixture's exact score is known in closed form, we can plant labelled
points around the modes, carry them back to seed space with the exact field, and fit
predictors of a seed's fate from them (the *atlas*). Each predictor is scored against the
fates the **learned** sampler actually produces.

## Layout

```
code/
  main.py          Hydra entry point: runs the stages
  train.py         stage 1: one learned sampler per (d, K, seed) + its ground-truth fates
  evaluate.py      stage 2: anchors -> exact backtrack -> predictors -> scores (one file per cell)
  visualize.py     stage 3: results.json -> (d, K) heatmaps and a LaTeX/PNG table
  core.py          numerics: GMM, score network, training recipe, exact sampler, anchors
  fate.py          the fate predictors (knn, altered_knn, quadratic, polar) and metrics
  processes/       ddim.py (VP diffusion, DDIM), flow.py (flow matching, OT path)
  runstate.py      named runs: settings check, invocation log
  combine.py       merge a run built on another machine into this one
  rank_probe.py    standalone: is the hallucination set low-rank?
  unit_tests/      python code/unit_tests/run.py
conf/              Hydra configs: config.yaml + one folder per group
scripts/main.sh    the full sweep, fanned out over all GPUs  (scripts/slurm/: the same on SLURM)
output/<run_id>/   everything a run produces (below)
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # tested: python 3.12, torch 2.5.1+cu121
```

## Run

```bash
python code/main.py                                   # the sweep in conf/sweep/base.yaml
python code/main.py sweep.d=[16,64] sweep.K=[4,8]     # another grid
python code/main.py process=flow                      # flow matching instead of DDIM
python code/main.py stages=[visualize]                # re-plot only
./scripts/main.sh                                     # the full grid on every GPU
DIMS="512" KS="2 4 8 16" PROCESSES=ddim ./scripts/main.sh
```

Every setting is a Hydra override (`conf/` holds the defaults). Predictors to run are the
`classifier@classifier.models.*` lines of `conf/config.yaml`.

## Runs are named and additive

A run is a **name** (`run_id`, default `abc123`); everything lives in `output/<run_id>/`:

```
run.json, history.log, invocations/        settings, one line / one config per invocation
checkpoints/<process>/<variant>/           checkpoints/model_d<d>_K<K>_s<seed>.pt, gt_cache/, manifest.json
<process>/<variant>/T<T_true>/             cells/d<d>_K<K>_s<seed>.json   (the result rows)
                                           results.json, results.csv, results_per_seed.csv
                                           table.tex, table.png, <model>.png, anchors_<b>/
logs/                                      scripts/main.sh job logs
```

`<variant>` is `unweighted` or `weighted` (`data.weighted`: random mixing weights). Every stage
skips what already exists, so re-running a run resumes it and a sweep split into pieces (other
d, more seeds, another anchor budget, another T_true or predictor) gives the files of one big
invocation. The first invocation records the settings in `run.json`; a later one that changes
how cells are computed (training recipe, eval size, a predictor's hyper-parameters, ...) is
refused (`strict_run=false` overrides). Use a new `run_id` for a new experiment.

**Several machines.** Run the same `run_id` elsewhere (e.g. `DIMS="512"`), copy that machine's
`output/<run_id>/` back, then

```bash
python code/combine.py /path/to/copied/abc123 --dry-run    # report first
python code/combine.py /path/to/copied/abc123              # merge + rebuild results and figures
```

It checks the settings match, copies what is missing, merges cell rows and manifests, and
never mixes a model with results computed from a different model of the same cell.

## Method notes

- **Ground truth** = fates of `eval.n_eval_per_mode * K` seeds under the learned sampler,
  cached at train time. Metrics: full accuracy, mode accuracy / macro F1, hallucination
  precision / recall / F1. Every number is mean ± std over `n_seeds` repeats (seeds 0, 100,
  200, ...), each with its own mode placement, model and eval seeds.
- **Training recipe** (`conf/train/base.yaml`): Adam, cosine lr decay, gradient clipping, EMA
  weights; retried with more steps while the probe hallucination rate exceeds `hall_target`.
  Width 256 for d ≤ 64, 1024 above (the R99 margin shrinks like 1/√d). Each seed has its own
  RNG, so training seeds together (one CUDA graph, ~15x faster) or apart gives the same model.
- **T_train = 500.** At 150 steps even the *exact* score hallucinates in high d (15/52/93/100 %
  at d = 32/64/128/256): a mode's 99% ball is a shell of relative width ~1.2/√d, so small
  discretisation error ejects samples. With T = 500 the exact score stays ≤ 0.6 % up to d = 256.
- **Exact field.** DDIM: closed-form GMM score, Heun steps (data → seed → data round-trips to
  the same label). Flow: the marginal OT velocity including the within-mode variance.
- **Anchors** (`conf/anchors/`): per mode, b points uniform in the R99 ball plus b/2 in the band
  R99 .. R99 + 2σ (hallucination class); altered_knn uses weighted concentric rings instead.
