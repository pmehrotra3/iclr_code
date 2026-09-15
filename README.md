# Seed-fate atlas — predicting a sampler's fate from the seed alone

Deterministic samplers (DDIM, flow-matching ODEs) map an initial Gaussian seed to a
data-space endpoint. On a Gaussian-mixture reference over a `(d, K)` sweep this repo asks:
**from the seed alone, can you tell which mode a sample will land in, or that it will
hallucinate (land outside every mode)?** — and how simple the seed → fate map is.

Each generative process lives in its own folder with its own Hydra entry point and config;
the shared library `code/common/` provides the reference GMM, the network backbone, the seed → fate
classifiers, and process-agnostic pipeline stages, so every process produces the same
results files, figures and tables.

```
code/
  common/            shared library
    gmm.py            K isotropic modes on a sphere; R99; fate labels; closed-form score
    nets.py           ScoreNet backbone (eps for ddim, velocity for flow)
    fate.py           seed -> fate classifiers: knn | kernel | altered_knn | linear | margin | radial | quadratic | poly | polar | mlp
    process.py        Process contract + registry (@register / load_process)
    checkpoint.py     checkpoint format shared by all processes
    stages/           train | evaluate | merge | seedmap | visualize   (all take cfg)
    conf/             shared Hydra groups: base.yaml, sweep/{full,ladder,quick,atlas},
                      classifier/{mlp,polar,linear,quadratic,knn,kernel,altered_knn,ladder,atlas}
  ddim/               VP diffusion + deterministic DDIM     (process.py, main.py, conf/config.yaml)
  flow/               OT flow matching, Euler/midpoint/RK4  (process.py, main.py, conf/config.yaml)
scripts/              run.sh (one process) | run_parallel.sh (split by d over GPUs) | run_all.sh | run_atlas.sh (ring-atlas T sweep)
checkpoints/<process>/    model_d{d}_K{K}.pt + manifest.json     (train)
output/<process>/         results[_tag].{json,csv}               (evaluate / merge)
visualization/<process>/  figures, seed maps, LaTeX tables        (visualize / seedmap)
```

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt and run from the repo root
export ATLAS_ROOT=$(pwd)    # where checkpoints/ output/ visualization/ go (defaults to $PWD)
```

## Run

```bash
python code/ddim/main.py                               # ddim: train -> evaluate -> visualize
python code/flow/main.py                               # same for flow matching
scripts/run_parallel.sh ddim                      # evaluate split by d over all GPUs, merge, plot
scripts/run_all.sh                                # both processes end to end
```

Every knob is a Hydra override; the process folders share the same ones:

```bash
# which stages
python code/ddim/main.py stages=[evaluate,visualize]                 # reuse checkpoints
python code/ddim/main.py stages=[visualize]                          # re-plot results.json only
python code/ddim/main.py stages=[seedmap] seedmap.K=[4,8,16]         # d=2 seed-space maps

# sweep and budgets
python code/ddim/main.py sweep=quick                                 # 2 cells, one small budget
python code/ddim/main.py sweep.d=[2,8,32] sweep.K=[8] sweep.budgets=[200000,2000000]

# which classifiers (the capacity ladder)
python code/ddim/main.py classifier=mlp                              # universal-approximator ceiling (default)
python code/ddim/main.py classifier=polar                            # degree-8 polynomial in (direction, radius) + margin
python code/ddim/main.py classifier=knn classifier._shared.k=25      # k-nearest-neighbour vote (no training)
python code/ddim/main.py classifier=kernel classifier._shared.bandwidth=2   # Gaussian-kernel vote
scripts/run_atlas.sh ddim classifier=altered_knn anchors.rings.n_rings=8    # weighted ring kNN, low confidence = hallucination
python code/ddim/main.py classifier=quadratic                        # quadratic boundaries (explicit + polar deg 2)
python code/ddim/main.py classifier=linear                           # hyperplanes (+ margin, + norm term)
python code/ddim/main.py classifier=ladder sweep=ladder              # linear .. polar1..8 .. mlp, all cells at 1M seeds
python code/ddim/main.py classifier.models.0.degree=4 classifier.primary=polar8
python code/ddim/main.py "classifier.models=[{name: p3, arch: polar, degree: 3}]" classifier.primary=p3

# whose fate is the ground truth
python code/ddim/main.py eval.labels=true eval.tag=true              # exact-score control: results_true.json,
                                                                #   figures under visualization/ddim/true/
# process-specific knobs
python code/ddim/main.py ddim.beta_max=15 train.force_retrain=true
python code/flow/main.py flow.solver=rk4 flow.sigma_min=0.001
python code/flow/main.py train.net.h=512 train.net.nb=6
```

`python code/<process>/main.py --cfg job` prints the fully composed config; `--help` lists the groups.

### Ring atlas: true-score backtrack → predictor, swept over T

```bash
scripts/run_atlas.sh                       # ddim then flow; each split by d over all GPUs
scripts/run_atlas.sh ddim                  # one process
scripts/run_atlas.sh ddim anchors.n_per_mode=1000 train.force_retrain=true
python code/ddim/main.py sweep=atlas classifier=atlas stages=[atlas,atlas_viz]   # single GPU
python code/ddim/main.py sweep=atlas classifier=atlas stages=[atlas_viz] run_tag=2026-09-15  # re-plot
```

Per `(d, K)`: the learned sampler (`sweep.T_train` = 150 steps, trained on demand until its
hallucination rate ≤ `train.hall_target` = 3 %, cached in `checkpoints/<process>/model_d{d}_K{K}_T150.pt`
until `train.force_retrain=true`) labels `eval.n_eval` = 20000 forward seeds as ground truth.
For every T in `sweep.T_atlas` (50 … 1000 step 50): `anchors.n_per_mode` = 500 data-space points
per mode uniform in the R99 ball (label = mode) plus `anchors.shell_frac` × 500 in the shell
R99 … (1 + `anchors.shell_w`) R99 (label = hallucination) are backtracked to seed space with the
**true** score at T steps; every predictor in `classifier=atlas` (knn, altered_knn, kernel, linear,
quadratic, polar2, polar3, polar8, mlp) is fit on those anchors and scored on the learned sampler's
20000 seeds.

`altered_knn` uses its own anchor set (`anchors.rings`, saved as `anchors/d{d}_K{K}_T{T}_rings.npz`):
per mode, `n_rings` concentric spheres of radius up to `r_max`·R99, every anchor labeled with its mode
and weighted by a radius-decaying confidence (`weight: linear | gaussian`) — **no hallucination class**.
A seed's k nearest anchors cast a weighted vote, `p = softmax(vote / temperature)`, and the seed is
declared a hallucination when the vote is low-confidence: `1 − H(p)/log K < threshold` (or `max p`
with `confidence: max`). `threshold: auto` picks the cut that maximises hallucination F1 on
`anchors.n_calibrate` seeds labeled by the **true** score at the same T (no learned-model information);
the chosen cut and its calibration F1 are stored per cell in `results.json` (`fit`). Outputs are dated:

```
output/<date>/<process>/T_<T>/results.{json,csv}      per-cell metrics at that T; summary.csv over T
output/<date>/<process>/T_<T>/tables.{tex,txt}        one (K x d) table per predictor family, cell =
                                                       overall acc / mode-basin F1 / hallucination F1 (%);
                                                       all_tables.{tex,txt} = every T in one file
output/<date>/<process>/anchors/d{d}_K{K}_T{T}.npz     the anchor set behind each number: P (data space),
                                                       y (labels), A (backtracked seeds), y_roundtrip; anchors.json
visualization/<date>/<process>/T_<T>/heatmap_full_acc  overall accuracy over the (d, K) grid
                                     heatmap_hall_f1   hallucination F1
                                     heatmap_mode_f1   mode-basin F1
                                     heatmap_roundtrip anchors that return to their label (sanity)
                                     anchors_K{K}      d = 2: the anchors in data space and in seed space
visualization/<date>/<process>/summary_vs_T, table_atlas.tex
```

Metric definitions (`common/fate.py: fate_metrics`): **overall accuracy** — fraction of held-out
seeds whose fate (mode k or hallucination) is predicted exactly; **hallucination F1** — one-vs-rest
F1 of the hallucination class; **mode-basin F1** — macro average over the K modes of the one-vs-rest
F1 of "predicted mode k" vs "truly mode k" (a hallucinating seed predicted as k counts against k).

## The pipeline

| stage | what it does | writes |
|---|---|---|
| `train` | one model per `(d, K)`; retrained with more steps until its hallucination rate ≤ `train.hall_target` | `checkpoints/<process>/model_d{d}_K{K}.pt` |
| `evaluate` | label held-out seeds with the ground-truth sampler; score the **analytic-field** forward pass and every classifier in `classifier.models` at every budget in `sweep.budgets` | `output/<process>/results[_tag].{json,csv}` |
| `merge` | combine `output/<process>/_parts/*/` (from `run_parallel.sh`) into one results file | same |
| `seedmap` | d = 2 fate maps: truth, the other sampler, every classifier with its errors in red | `visualization/<process>/seedmap_K{K}.{png,pdf}` |
| `visualize` | accuracy vs d / budget, hallucination F1, heatmaps, capacity ladder, LaTeX tables | `visualization/<process>[/tag]/` |
| `atlas` | ring atlas over T (see below); `atlas_merge` combines parallel parts, `atlas_viz` draws the per-T heatmaps | `output/<date>/<process>/T_<T>/`, `visualization/<date>/<process>/T_<T>/` |

**Ground truth** (`eval.labels`): `learned` (default) — the trained model's sampler; `true` — the
analytic-field sampler (exact score / exact OT velocity), the control that shows how predictable
fate is when the model is perfect.

**Predictors**: `analytic` — run the analytic-field sampler on the seed and read off the mode
(no training; cannot see the learned model's errors). `classifier.models` — trained on seeds
labeled by the ground-truth sampler; `arch` fixes the boundary geometry (see `common/fate.py`):
`knn` / `kernel` (non-parametric votes over the labeled seeds) · `linear` (hyperplanes) → `radial`
(+‖x‖²) → `quadratic` → `polar` (degree-p polynomial in direction × radius, hallucination = margin
band around a mode boundary) → `mlp` (universal). Presets: `classifier=mlp|polar|linear|quadratic|
knn|kernel|ladder|atlas`; any list of `{name, arch, …}` entries works inline.

## Adding a process

Create `code/<name>/process.py` with a class decorated `@register("<name>")` implementing
`build_model`, `train_model`, `sample`, `true_forward` (and optionally `extra_ckpt`), copy
`code/ddim/main.py` and `code/ddim/conf/config.yaml`, set `process: <name>` and add any
process-specific knobs. Nothing else changes — the stages, figures and tables are process-agnostic.

## Notes

- **Noise schedule (ddim).** The continuous VP schedule keeps ᾱ_T ≈ 4e-5 for any `T`; the
  discrete DDPM `linspace(1e-4, 0.02, T)` schedule only reaches pure noise at `T = 1000` and at
  `T = 100` leaves ᾱ_T ≈ 0.36, which silently corrupts the seed → fate map. Checkpoints carry a
  `schedule` tag and are retrained if it is stale.
- **Budgets.** Labeling costs ≈ 10 s per 1M seeds on one GPU; the MLP ensemble ≈ 45 s per budget,
  the polynomial families ≈ 20 s. The full grid × 3 budgets is ≈ 1 h per process on 4 GPUs.
- **Reproducibility.** Seeds are fixed (`seed`, `eval.seed_offset`); the same seeds are used for
  every classifier so ladders compare like with like.
