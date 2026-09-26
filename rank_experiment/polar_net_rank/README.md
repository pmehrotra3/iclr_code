# Projection rank: the polar network on one pulled-back sphere

**Question.** The polar network (`code/fate.py`, `FateNet` with `arch="polar"`) starts with
`r = hidden = 256` linear projections of a seed `x` in noise space,

```
z_j(x) = (w_j . [x/|x| sqrt(mu), |x| - mu] + b_j) / sqrt(mu + 1),     mu = sqrt(d),
```

and only then applies its nonlinearity, the powers `(z, z^2, z^3)`. Take `r` distinct points,
all on one of the concentric spheres around a mode, pulled back to noise space. Before the
nonlinearity, can the `r` projections tell those `r` points apart? That is, is the `r x r`
matrix `z_j(x_i)` of full rank?

**What is measured.** For DDIM and flow matching, and every `d` in `2 ... 512` and `K` in `2, 4, 8`:

1. The Gaussian mixture of the main experiments (first run, seed 0, unweighted) and its polar
   network, built with `code/`'s own `FateNet` in two versions: at initialisation, and trained as
   in the main experiments. The trained version is ensemble member 0, trained on 20,000 anchors
   per mode pulled back by the exact field with `T_true = 500`.
2. The sphere of radius `R99` around mode 0 in data space. Draw `r` points on it and pull them
   back to noise space with the exact map (the same map that pulls back the anchors). The
   result is `r` distinct points, all on the same pulled-back sphere.
3. The **same** `r` points go through **all** `r` projections, which gives one `r x r` matrix
   `H[i, j] = z_j(x_i)`. The table reports its rank (before the nonlinearity) and the rank of
   `[H, H^2, H^3]` (after it). Both are computed in float64, and the script checks them against
   the network's own float32 projections.

Before the nonlinearity every column is affine in the `d + 1` inputs (direction and radius), so
`rank H <= min(r, d + 2)`. That is below `r = 256` for every `d <= 128`, whatever the weights.

## Run it

```bash
./rank_experiment/polar_net_rank/main.sh quick    # d <= 32, 2,000 anchors per mode
./rank_experiment/polar_net_rank/main.sh          # the full grid of config.yaml
./rank_experiment/polar_net_rank/main.sh tables   # rebuild the .tex and .png from the saved CSVs
```

Every setting is in [config.yaml](config.yaml): the sphere's radius (in units of `R99`; the
w-kNN rings are 0.3 to 1.5), the mode, and the number of points (default `r`).

## Output

```
rank_experiment/polar_net_rank/abc/
├── ddim/    rank.csv  rank.tex  rank.png
└── flow/    rank.csv  rank.tex  rank.png
```

One row per `(K, d)`. The table shows the number of projections and of points, the bound
`min(r, d + 2)`, and the rank before and after the nonlinearity, at initialisation and trained.
The `.csv` also holds a few checks:

- the smallest distance between two of the points (all distinct);
- the singular-value gap at each rank cut (largest dropped over smallest kept; about `1e-15`
  means the rank is unambiguous);
- the training time.
