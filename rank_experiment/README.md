# Rank experiment: do hallucinating points have structure?

**Question.** Seeds that make the sampler hallucinate: are they spread over all of the noise
space like ordinary Gaussian noise, or do they occupy only a few dimensions? The same question
is asked of the samples they produce (data space). It is asked for both processes of the
paper, **DDIM** and **flow matching**.

**Answer we measure.** For each process and every dimension `d` and number of modes `K`:

1. Build the Gaussian mixture of the main experiments (centres on a sphere of radius
   `2 sqrt(d/2)`, at least `3 R99` apart, component std `0.1`).
2. Send fresh seeds through the process's **exact** map (no trained network), the same map as
   in `code/`: the exact-score DDIM map (`core.forward_true`) or the exact-velocity
   flow-matching map (`processes/flow.py`, optimal-transport path). Keep the seeds whose sample
   lands outside every `R99` ball: the **hallucinating seeds**. Both processes start from the
   same seeds.
3. Compare three sets with the same number of points:

   | set | what it is |
   |---|---|
   | **points in noise space that would hallucinate** | the hallucinating seeds |
   | **points that hallucinate in data space** | the samples they produce |
   | **points sampled from Gaussian noise** | fresh standard normal points: the reference, with no structure |

4. Measure how many dimensions each set really uses, after removing its mean:
   - **effective rank**: `exp(entropy)` of the normalised singular values;
   - **dimensions for 99% of the variance**: principal components needed to hold 99% of it.

   Points sampled from Gaussian noise give about `d` for both. A structured set gives much less.

Both measures are computed in two coordinate systems, **Euclidean** and **polar**
(hyperspherical: radius and `d - 1` angles), from the same hallucinating seeds.

## Run it

```bash
./rank_experiment/main.sh quick     # small grid (d <= 32, K <= 8), a few minutes on a laptop
./rank_experiment/main.sh           # full grid of config.yaml (d up to 1024, K up to 32), both processes
./rank_experiment/main.sh tables    # rebuild the .tex and .png from the saved CSVs
./rank_experiment/main.sh full --process flow    # one process only (ddim | flow)
```

Every setting is in [config.yaml](config.yaml). The device is chosen automatically (NVIDIA GPU,
else Apple GPU, else CPU); force one with `./rank_experiment/main.sh full --device cpu`. An
interrupted run resumes where it stopped. Needs `numpy`, `torch`, `matplotlib`, `pyyaml`.

Both runs write to `rank_experiment/abc/`. After changing any setting other than the grid,
delete `rank_experiment/abc/` first, so that every cell is recomputed. A cell counts as done once it has the `n_hall` points
the run asks for, so the full run redoes the cells of a quick run (2,000 points instead of
20,000; their rows are dropped when the full run starts, so a table never mixes the two), and
a quick run after a full run leaves the full results alone.

## Output

One folder per process, and in it one table per coordinate system, each in three formats:

```
rank_experiment/abc/
├── ddim/    euclidean.csv  euclidean.tex  euclidean.png  polar.csv  polar.tex  polar.png
└── flow/    euclidean.csv  euclidean.tex  euclidean.png  polar.csv  polar.tex  polar.png
```

Each table has one row per `(K, d)`: the hallucination rate (%), the number of points in each
set, and the effective rank and the dimensions for 99% of the variance of the points in noise
space that would hallucinate, the points that hallucinate in data space and the points sampled
from Gaussian noise. The `.csv` holds the numbers, the `.tex` is ready for the paper (needs `booktabs` and `graphicx`;
it shrinks to the text width only when it is wider), and the
`.png` is the same table as an image.

## Notes

- The seeds are drawn on the CPU, so a run gives the same seeds on any device, and both
  processes see the same seeds.
- `rank_sweep.py` writes both maps with the algebra rearranged, so that a step reads each
  large array a few times instead of about fifty (4-5 times faster). On 20,000 seeds at
  `d = 64` and `d = 1024` (`K = 8`) it gives the same hallucinating seeds as `code/`, and
  samples within `5e-3` of it.
- DDIM's `beta_min` is `1e-4`, as in the main experiments. Like the main experiments' sampler, the map
  stops at the least noisy level `abar_0 = 1 - beta_min`, so its samples keep noise of std
  `sqrt(beta_min)` in every coordinate. With the original notebook's `1e-3` that noise alone
  pushes samples just outside the `R99` ball in high `d` (53% of them at `d = 1024`, `K = 8`,
  against 3% with `1e-4`), so the table would measure the noise instead of hallucinations.
- Even the exact sampler lands outside the `R99` balls about `1 - mass_q = 1%` of the time,
  because `R99` holds 99% of each component's mass.
- `code/rank_probe.py` is the earlier single-cell version of this probe.
