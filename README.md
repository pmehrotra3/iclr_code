# Seed-fate atlas — where does a seed go, according to the exact score?

A deterministic sampler (DDIM, a flow-matching ODE) maps a Gaussian seed to a data-space
endpoint. On a Gaussian-mixture reference over a `(d, K)` grid this repo asks: **can the seed's
fate — which mode it lands in, or that it hallucinates — be read off a picture of seed space
built from the exact score alone, with no model in the loop?** The trained network is used
once, to produce the ground truth the picture is scored against.

```
code/
  common/            shared library
    gmm.py            K isotropic modes on a sphere; R99; the labelling rule L; anchors; closed-form score
    nets.py           ScoreNet backbone (eps for ddim, velocity for flow)
    process.py        Process contract + registry; Process.fit = the training recipe (EMA, cosine lr, clip)
    fate.py           the predictor family: knn | kernel | altered_knn | linear | quadratic | polar | mlp
    checkpoint.py     checkpoint format (carries the training recipe; stale checkpoints retrain)
    stages/           train | atlas | atlas_merge | atlas_viz
    conf/             base.yaml, sweep/{full,d2,quick}, classifier/{atlas,fast}
  ddim/               VP diffusion + deterministic DDIM        (process.py, main.py, conf/config.yaml)
  flow/               OT flow matching, deterministic ODE       (process.py, main.py, conf/config.yaml)
scripts/run.sh        the experiment: shard the grid by d over GPUs, merge, draw
checkpoints/<process>/model_d{d}_K{K}_T150.pt      the learned samplers (+ manifest.json)
output/[<run_tag>/]<process>/                       results, tables, figures
```

## Install and run

```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
export ATLAS_ROOT=$(pwd)

scripts/run.sh                                     # both processes, the full grid  (~hours on 4 GPUs)
scripts/run.sh ddim flow sweep=d2 classifier=fast  # d = 2, four predictors          (~20 min)
scripts/run.sh ddim stages=[atlas_viz]             # re-draw from output/ddim/atlas_results.json
python code/ddim/main.py sweep=quick classifier=fast anchors.budgets=[20000] eval.n_eval=20000   # smoke test
python code/ddim/main.py --cfg job                 # print the composed config
```

Every knob is a Hydra override (`sweep.K=[4,16] anchors.budgets=[50000] eval.n_eval=100000
run_tag=try1 data.sigma=0.2 ddim.true_solver=euler ...`); `code/common/conf/base.yaml` documents them.

## The method

Per `(d, K)` cell (`code/common/stages/atlas.py`):

**Step 0 — Reference mixture.** K isotropic Gaussians of std σ, centres on the sphere of radius 2.
R99 is the radius holding 99 % of a mode's mass. The labelling rule **L**: a point belongs to mode
k if μ_k is its nearest centre and lies within R99; otherwise it is a *hallucination* (−1).

**Step 1 — Plant labelled anchors in data space.** No model. Around each centre: points uniform in
the R99 ball, plus half as many uniform in radius over the band R99 … R99 + 2σ just outside it —
the hypothesis under test, that hallucinations live in a thin band around each mode. Every anchor
is coloured with L (`gmm.ball_anchors`). Budgets `anchors.budgets` = 20k / 50k / 100k anchors per
cell; every budget is fitted and the best is reported.

**Step 2 — Backtrack every anchor to seed space with the exact score.** Reverse DDIM (or the
reverse OT flow) for T steps driven by the closed-form score of the noised mixture, carrying the
colour. A Heun predictor–corrector step (`ddim.true_solver`, `flow.true_solver`) makes the
backward and forward passes exact inverses: the *roundtrip* check (backtrack, push forward, compare
L) is 1.000. T ∈ `sweep.T_atlas` = 100 / 200 / 500 / 1000.

**Step 3 — Fit predictors on the labelled seed picture, and nothing else.** Seeds enter as
φ(z) = (z/‖z‖, ‖z‖) where the family uses it. The family (`classifier=atlas`, `common/fate.py`):
kNN and Gaussian-kernel votes; linear; quadratic; polynomials in (direction, radius) of degree 2, 3
and 8; a depth-4 MLP ceiling. Each parametric model is also reported **prior-calibrated**
(`<name>_cal`): its hallucination cut is shifted so it calls exactly as many seeds hallucinations
as the exact score does at that T (measured on `anchors.n_calibrate` exact-score-labelled seeds).
`altered_knn` differs in kind: it trains on mode-labelled anchors only (`gmm.altered_knn_anchors`,
concentric spheres with radius-decaying weights) and declares a hallucination when the k-nearest
vote is low-confidence (`1 − H(p)/log K < threshold`, threshold chosen on exact-score labels), so it
tests whether hallucination regions are identifiable as "nowhere in particular". Reported separately.

**Step 4 — Ground truth from the learned sampler.** `eval.n_eval` = 200 000 fresh Gaussian seeds
pushed through the trained network Φθ with its own plain sampler at `sweep.T_train` = 150 steps
and labelled with L. The only place the model is used.

**Step 5 — Score, and the analytic control.** Each predictor labels the same seeds; report overall
accuracy, mode accuracy / mode-basin F1, hallucination precision / recall / F1. The control pushes
the seeds through the exact score — at the atlas T (`analytic`) and at T_train
(`analytic_T_train`). **The control is a ceiling, not a baseline**: it says how much of the learned
sampler's behaviour the exact score explains at all; no predictor built from the exact score can
beat it, and a cell where it scores badly is measuring model error, not predictor quality.
Diagnostic `hall_dist`: where the sampler's real hallucinations land, in σ beyond R99 (98–100 %
within 2σ at d = 2 — the band hypothesis holds).

**Step 6 — Sweep** over the `(d, K)` grid, T and the anchor budget (`sweep=full`).

### The learned sampler (`stages/train.py`, `Process.fit`)

One ScoreNet (h = 256, 4 blocks) per cell, denoising-score-matching / flow-matching loss,
30 000·(1 + d/16)(1 + K/16) steps of Adam with **cosine lr decay to 1 %, gradient clipping at 1
and an EMA of the weights (0.999)**; retrained with 1.7× more steps (≤ 4 attempts) until its
hallucination rate is ≤ 1 % + 0.5 % (1 % = 1 − `data.mass_q`, the mass a *perfect* sampler leaves
outside the R99 balls). Checkpoints store the recipe and retrain on demand when it changes.

### Outputs

```
output/<process>/atlas_results.json      every cell, T, budget, predictor (the figures read this)
output/<process>/summary.csv             mean over cells per (T, predictor) at the best budget
output/<process>/T_<T>/results.csv       one row per (cell, predictor, budget): metrics, roundtrip_acc
output/<process>/T_<T>/<model>/heatmap_{full_acc,mode_f1,hall_f1}   (K x d) grids, winning budget in brackets
output/<process>/T_<T>/summary_all_models_<metric>                  every predictor side by side
output/<process>/summary_vs_T.{png,pdf}  mean metrics vs T, every predictor and the control
output/<process>/best_T_table.{tex,png}  best T per predictor and the full grid at the best (predictor, T)
```

Metrics (`fate.fate_metrics`): **overall accuracy** — seeds whose fate (mode k or hallucination) is
predicted exactly; **mode-basin F1** — macro one-vs-rest F1 over modes (a hallucinating seed
predicted as k counts against k); **hallucination F1** — one-vs-rest F1 of the hallucination class.

## Results at d = 2 (σ = 0.1, T = 500, 200k seeds; `output/ddim`, `output/flow`)

| K | ddim ceiling | ddim polar8_cal | flow ceiling | flow polar8_cal |
|---|---|---|---|---|
| 2 | 0.990 | 0.990 | 0.995 | 0.995 |
| 4 | 0.984 | 0.983 | 0.989 | 0.988 |
| 8 | 0.973 | 0.971 | 0.981 | 0.978 |
| 16 | 0.962 | 0.962 | 0.980 | 0.978 |

Every model-free predictor sits within ~0.3 pt of the analytic ceiling; the residual (0.5–4 %,
growing with K) is the learned sampler's own disagreement with the exact score, concentrated in a
thin band along the basin boundaries.

## Notes

- **Anchors must be coloured by L, not by the planting mode.** At d = 2 centres are only forced
  3σ apart while R99 = 3.03σ, so balls overlap; colouring by planting mode mislabels 15–50 % of the
  anchors before any dynamics and costs up to 8 pts of accuracy.
- **Noise schedule (ddim).** Continuous VP, ᾱ_T ≈ 4e-5 for any T; checkpoints carry a schedule tag.
- **Costs.** Heun doubles the exact-score passes; the d = 2 grid is ~20 min per process, the full
  grid a few hours on 4 GPUs, dominated by training at d = 32.
- **Adding a process.** `code/<name>/process.py` with `@register("<name>")` implementing
  `build_model`, `train_model` (call `self.fit`), `sample`, `true_forward`, `true_backward`; copy
  `code/ddim/main.py` and `conf/config.yaml`. Stages and figures are process-agnostic.
