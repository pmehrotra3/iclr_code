# animations

2-D animations of the Gaussian-mode experiment (d = 2, K = 4, the seed-0 models), made from the
trained checkpoints in `checkpoints/`, for every sampler and both variants.

```bash
python animations/make_animations.py                                   # all 5 samplers, unweighted + weighted
python animations/make_animations.py --procs ddim --variants weighted  # one sampler / variant
python animations/make_animations.py --procs flow --K 8 --frames       # another K, every frame as PNG
```

## Layout

```
out/<sampler>/<variant>/<sampler>_<variant>_K4_d2_<kind>.gif
    sampler: ddim, flow, heun, rk45, dpmpp2m        variant: unweighted, weighted
```

| kind | samplers | what it shows |
|---|---|---|
| `learned` | all 5 | the **learned score**: seeds x_T ~ N(0, I) are sampled to x_0, then played back **from the generated samples x_0 to the noise x_T they came from**. Each point keeps the colour of the mode it lands in; hallucinations are red ×, so each one can be followed back to its seed (the basin boundaries, plus a few far-out seeds). |
| `true` | ddim, flow | the same with the **true score**. |
| `anchors` | ddim, flow | the **anchor points** that kNN, quadratic and polar are fit on: uniform inside each mode's 99% circle (labelled with that mode) plus the shell just outside it, R99 to R99 + 2σ (labelled hallucination), **pulled back with the true score from data space to noise**. |

Heun, RK45 and DPM-Solver++(2M) only have `learned`. They integrate DDIM's network, and their true
score and anchor points are DDIM's, so those two animations would be the same as DDIM's.

## Conventions (keep them the same in the paper's figures)

- **Hallucination:** red (`#D62728`) ×, and nothing else is red. The mode colours
  (`MODE_COLORS`: blue, green, purple, amber, …) contain no red.
- **Modes:** solid circle = the mode's 99% circle (radius R99). In the anchor animation, the
  hallucination shell is shaded red with a dashed outer edge.
- **Step counter:** the same for every sampler. Step 0 is the data end (a *generated sample* x_0,
  or the anchor points where they are placed), and step T−1 = 499 is the noise x_T.
- **Names:** "learned score", "true score", "anchor points", "pulled back", and "99% circle". To
  rename anchor points everywhere, change `ANCHOR` in the script.
- **The view** starts zoomed on the modes and zooms out to the noise [−3.6, 3.6]² as the points
  travel. The mode circles fade out on the way, since they only mean something in data space.

## How the counts work

Only about 1% of generated samples hallucinate: that is roughly the rate outside R99 even for a
perfect model. With 5,000 seeds you'd see only about 55, too few to trace the boundaries. So
`learned` and `true` sample a pool of 80,000 seeds (`--pool`). They show every hallucination
(up to `--max-hall 1000`) and a random 4,000 of the other samples (`--show`), and the title
states both counts and the true rate.

The anchor animation uses the best anchor budget in `output/abc123/<sampler>/<variant>/T500/results.json`
for this (d, K), which is 20,000 per mode at d = 2, K = 4, or `--anchors N`. It shows a random 400
per mode inside the circles and 200 in the shells (`--show-anchors`). Each point's path doesn't
depend on the others, so the paths shown are exactly those of the full set.

Other options: `--seed 100|200` (another trained model), `--every` (steps per frame), `--fps`,
`--frames` (every frame as a PNG, for making a video: `ffmpeg -r 20 -i %04d.png out.mp4`),
`--forward` (play the seed animations noise → data). CPU only, about 35 minutes for everything,
most of it RK45 (6 network calls per step).
