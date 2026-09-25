# animations

2-D animations of the Gaussian-mode experiment (default d = 2, K = 4, the seed-0 models), for
DDIM and flow matching, made from the trained checkpoints in `checkpoints/`. All animations run
data → noise (`--forward` plays the seed ones noise → data); both fields use the learned sampler's T.

```bash
python animations/make_animations.py                                   # ddim + flow, unweighted + weighted
python animations/make_animations.py --procs flow --K 8 --frames      # another K, every frame as PNG
```

`out/` gets, per variant and process:

| file | what it shows |
|---|---|
| `<proc>_..._learned.gif` | the **learned** network: 5 000 seeds x_T ~ N(0, I) are sampled to x_0, then played back **from data to noise**. Each point is coloured by where it landed (a mode, or black × = hallucination), so every hallucination can be followed from x_0 back to the seed it came from (the basin boundaries, plus a few far-out seeds). Left: the samples x_0 in data space by fate. |
| `<proc>_..._exact.gif` | the same, with the **exact (true)** field instead of the network. |
| `<proc>_..._anchors.gif` | the **anchors**, placed as evaluate.py does (uniform in each mode's R99 ball, plus the band just outside it = hallucination class), backtracked with the exact field from the balls **out to noise**: the band anchors end on the basin boundaries. Left: the anchors in data space. |
| `frames/` | with `--frames`: every frame as a PNG, to make a video (e.g. `ffmpeg -r 20 -i %04d.png out.mp4`). |

Options: `--variants unweighted` (or `weighted`) for one variant only, `--seed 100|200` (another trained model), `--n` seeds,
`--anchors` per mode, `--every` steps per frame, `--fps`. CPU only, about a minute in total.
