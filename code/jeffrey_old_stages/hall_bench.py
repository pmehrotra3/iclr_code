"""hall_bench.py — how many ground-truth hallucinations does each method remove? (DDIM or flow)

Checkpoints and ground truth are the repo's own, made by stages=[train]:
checkpoints/ddim/<variant>/checkpoints/model_d<d>_K<K>_s<seed>.pt, sampled on the schedule and
mixing weights stored in the checkpoint, and gt_cache/d<d>_K<K>_s<seed>.pt, whose labels are used
as the ground truth after an agreement check (seed = bench.ckpt_seed, default cfg.seed;
<variant> follows data.weighted).

For every cell of bench.d x bench.K:

  1. Ground truth. Take the seeds whose ground-truth fate is "hallucinated" (label -1):
       bench.gt_source = cache   : the cell's gt_cache (fates of proc.seeds(n_eval, d, seed + 1)
                                   under the learned sampler, from train.py);
                         learned : run the learned sampler on bench.n seeds and label the
                                   endpoints (nearest mode within R99, else -1);
                         true    : the same with the exact-GMM field.
     If the cache is missing or unreadable, it falls back to `learned` and says so.
  2. Apply four repairs to exactly those seeds, each with the learned sampler, and count how
     many still land outside every core:
       ours     : move the seed by a fixed eps along -n, n = J^T nu / ||J^T nu|| (Prop. 4, one
                  VJP through the whole sampler), nu toward the nearest mode of its endpoint;
       IQ       : score s - lam grad E, E = DSM loss at t0 of the Tweedie estimate, applied only
                  for t <= bench.iq_window ("for small t", default 0.2);
       RODS-SAS : at each step in the window, evaluate eps at x + delta, delta = -rho s/||s||;
       RODS-CAS : the same with delta = rho grad||s|| / ||grad||s|| ||  (RODS, Tian et al. 2025,
                  eq. 8 and App. B.4; correction gated by the curvature index H if
                  bench.rods_thresh > 0, otherwise applied at every step in the window).
     Each method's one knob (eps, lam, rho) is chosen from its grid on bench.tune_n of the
     seeds, then the chosen value is run on all of them.

Only ground-truth hallucinations are touched, so this is a repair benchmark for every method.

Flow matching: process=flow benchmarks the flow checkpoints (checkpoints/flow/<variant>/) with the
same four methods; samplers.py translates IQ and RODS through the exact velocity <-> score
relations of the OT path, and every setting (eps, lam, rho, windows as the last w of the run)
means the same for both samplers.
Writes output/<run_id>/ddim/<variant>/hall_bench/{summary.csv,cells.json} after every cell, and
skips cells already in cells.json, so re-running the same command resumes a crashed run.
bench.grad_chunk bounds the seeds per autograd pass (the normal backprops through every step).
Stage hall_bench_viz draws the table.

Trajectories (bench.traj.save): every state of every hallucinated seed's trajectory, for the
original sampler and each method in bench.traj.methods, is cached while the cell runs:
hall_bench/traj/d<d>_K<K>/<method>.npy, shape (n_hall, T, d), [:, 0] the start and [:, -1] the
final sample, no step skipped; meta.npz holds the seeds, normals, classes, final fates and knobs.
Stage hall_bench_traj caches them for a finished run (same seeds, the knobs it chose, no tuning),
and load_traj(cell_dir) reads a cell back.

    python code/baselines/main.py 'stages=[hall_bench,hall_bench_viz]'
    python code/baselines/main.py 'stages=[hall_bench]' 'bench.d=[2]' 'bench.K=[2]' run_id=bench_test
"""
from __future__ import annotations
import os
import csv
import json
import shutil
import time
import numpy as np
import torch

import core
from train import gt_cache_path, variant_of
from .pullback_iq import (pulled_normal, level, load_cell, ckpt_process, ckpt_seed, stage_dir,
                          C_OG, C_IQ, C_US)
from .samplers import make_sampler, iq_energy, sample_iq


def bench_dir(cfg):
    return stage_dir(cfg, cfg.bench, "hall_bench")


def fig_dir(cfg):
    return os.path.join(bench_dir(cfg), "figures")


# ------------------------------------------------------------------ ground truth
def ground_truth(cfg, d, K, S, S_true, means_t, R99, device, proc=None):
    """Seeds whose ground-truth fate is -1, plus where that ground truth came from.

    The gt_cache holds labels for proc.seeds(n_eval, d, seed + 1), drawn by Process.seeds (what
    train/evaluate do), so the seeds are regenerated the same way and checked against the model
    before the labels are trusted."""
    b = cfg.bench
    src = str(b.gt_source)
    cap = None if b.get("n", None) in (None, "null", 0) else int(b.n)
    lab = lambda Z: torch.cat([core.label_fate(S.G(c), means_t, R99) for c in Z.split(int(b.chunk))])
    if src == "cache":
        path = gt_cache_path(cfg.paths.checkpoints, ckpt_process(cfg, b), d, K, ckpt_seed(cfg, b),
                             variant_of(cfg))
        gt = None
        try:
            gt = torch.load(path, map_location="cpu", weights_only=False)
            L = gt["gt"].long().reshape(-1)
            n_eval, seed = int(gt["n_eval"]), int(gt["seed"])
            Z = proc.seeds(n_eval, d, seed + 1)              # exactly train.py's eval seeds
            n = n_eval if cap is None else min(cap, n_eval)
            Z, L = Z[:n].contiguous(), L[:n].to(device)
            n_gt = n
            # Full labels (mode index or -1), not just hallucinated-or-not: with ~1% hallucinating,
            # a hallucination-only comparison agrees ~98% even for unrelated seeds.
            # Check seeds spread over the WHOLE set, not just the first ones: GPU random numbers are
            # generated in parallel blocks, so the first few thousand seeds can reproduce exactly on
            # another machine while the rest do not. Hallucinated seeds are checked on their own too.
            m = min(int(b.get("check_n", 5000)), n)
            gck = torch.Generator(device="cpu").manual_seed(12345)
            idx = torch.randperm(n, generator=gck)[:m].to(Z.device)
            agree = (lab(Z[idx]) == L[idx]).float().mean().item()
            hidx = torch.nonzero(L < 0).flatten()
            hidx = hidx[torch.randperm(hidx.numel(), generator=gck)[:min(1000, hidx.numel())].to(hidx.device)]
            h_agree = (lab(Z[hidx]) < 0).float().mean().item() if hidx.numel() else 1.0
            print(f"[hall_bench]   gt vs the sampler: {100*agree:.2f}% of labels agree on {m} random seeds; "
                  f"{100*h_agree:.1f}% of the cached hallucinated seeds hallucinate here")
            agree = min(agree, h_agree)
            if agree < float(b.get("min_agree", 0.9)):
                raise ValueError(f"only {100*agree:.1f}% of labels agree: seeds not reproduced")
            del gt
            how = f"gt_cache:{os.path.basename(path)} ({n_gt} seeds)"
            if cap is not None and n_gt < cap:                  # top up to bench.n seeds
                extra = cap - n_gt
                Zx = proc.seeds(extra, d, seed + 10007)
                Z, L = torch.cat([Z, Zx]), torch.cat([L, lab(Zx)])
                how += f" + {extra} model-labelled"
            return Z, L, how
        except Exception as e:
            print(f"[hall_bench]   gt cache unusable ({path}): {e}. Labelling with the sampler instead.")
            gt = None
            src = "learned"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    n = cap or 200000
    Z = proc.seeds(n, d, int(b.seed) + 17)
    SS = S_true if src == "true" else S
    L = torch.cat([core.label_fate(SS.G(c), means_t, R99) for c in Z.split(int(b.chunk))])
    return Z, L, f"{src}-sampler on {n} seeds"


def tube_class(X, means_t, R99):
    """Split hallucinated endpoints into 'interpolation' (inside the R99 tube joining some pair
    of modes, strictly between them) and 'invalid' (outside every core and every tube).

    For every pair (i, j): s = <x - mu_i, mu_j - mu_i> / ||mu_j - mu_i||^2 is the position along
    the segment and the distance to the segment is measured at clamp(s, 0, 1); a point counts as
    an interpolation when some pair has 0 < s < 1 and that distance <= R99. Returns the boolean
    mask, the distance to the nearest tube axis and the pair it belongs to."""
    K = means_t.shape[0]
    ii, jj = torch.triu_indices(K, K, offset=1)
    mi, mj = means_t[ii], means_t[jj]                        # (P, d)
    v = mj - mi
    vv = (v * v).sum(1).clamp_min(1e-12)
    s_ = ((X[:, None, :] - mi[None]) * v[None]).sum(-1) / vv[None]        # (B, P)
    inside = (s_ > 0) & (s_ < 1)
    proj = mi[None] + s_.clamp(0, 1)[..., None] * v[None]
    dist = (X[:, None, :] - proj).norm(dim=-1)               # (B, P)
    dist_in = torch.where(inside, dist, torch.full_like(dist, float("inf")))
    dmin, pair = dist_in.min(1)
    return dmin <= R99, dmin, pair


# ------------------------------------------------------------------ RODS
def grad_score_norm(S, x, i):
    x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        (g,) = torch.autograd.grad(S.score(x, i).norm(dim=1).sum(), x)
    return g


def sample_rods(S, z, rho, kind, thresh, window, chunk, record=None):
    """RODS-SAS / RODS-CAS on the DDIM sampler; correction only in the step window (fractions
    of the run, 0 = first step). record(traj) gets every chunk's states, (T, B, d), if given."""
    T = S.T
    lo, hi = float(window[0]), float(window[1])
    out = []
    for x in z.split(chunk):
        xs = [x] if record is not None else None
        for k, i in enumerate(range(T - 1, 0, -1)):
            frac = k / max(1, T - 2)
            if rho <= 0 or not (lo <= frac <= hi):
                with torch.no_grad():
                    x = S.step(x, i)
                if xs is not None:
                    xs.append(x)
                continue
            gn = grad_score_norm(S, x, i)
            u = gn / gn.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if thresh > 0:                                   # curvature index, eq. (8)
                H = (grad_score_norm(S, x + rho * u, i) - gn).norm(dim=1)
                on = H >= thresh
            else:
                on = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            with torch.no_grad():
                if kind == "sas":
                    s = S.score(x, i)
                    delta = -rho * s / s.norm(dim=1, keepdim=True).clamp_min(1e-12)
                else:                                        # cas
                    delta = rho * u
                e_hat = S.pred(x + delta, i)                 # the prediction at the moved point
                e = torch.where(on[:, None], e_hat, S.pred(x, i))
                x = S.update(x, i, e)
            if xs is not None:
                xs.append(x)
        if xs is not None:
            record(torch.stack(xs))
        out.append(x)
    return torch.cat(out)


# ------------------------------------------------------------------ trajectory cache
def traj_dir(cfg, d, K):
    return os.path.join(bench_dir(cfg), "traj", f"d{d}_K{K}")


class TrajCache:
    """Every state of every hallucinated seed's trajectory, one .npy per method.

    <traj_dir>/<name>.npy has shape (n_hall, T, d): along axis 1, row k is level T-1-k, so [:, 0]
    is where sampling starts (the seed; the moved seed for ours) and [:, -1] the final sample.
    Seeds are in the order of meta.npz["seed_index"]. Chunks are written straight to disk as
    they are produced (memory-mapped), and meta.npz / meta.json are written last, so a cell
    without them is incomplete. Names: original, ours_eps<e>, iq_t<w>, rods_sas, rods_cas."""

    FAMILIES = ("original", "ours", "iq", "rods")

    def __init__(self, cfg, d, K, n, T):
        c = cfg.bench.get("traj", None) or {}
        self.methods = {str(m) for m in (c.get("methods", None) or ["original", "ours"])}
        bad = self.methods - set(self.FAMILIES)
        if bad:
            raise ValueError(f"bench.traj.methods: unknown {sorted(bad)}; choose from {self.FAMILIES}")
        self.dtype = np.dtype(str(c.get("dtype", "float32")))
        self.dir = traj_dir(cfg, d, K)
        self.shape = (int(n), int(T), int(d))
        self._mm = {}

    def wants(self, family):
        return family in self.methods

    def check_disk(self, n_sets):
        need = int(np.prod(self.shape)) * self.dtype.itemsize * n_sets
        os.makedirs(self.dir, exist_ok=True)
        free = shutil.disk_usage(self.dir).free
        if free < 1.05 * need + 2 ** 30:
            raise RuntimeError(
                f"[hall_bench] the trajectories of this cell need {need / 2**30:.2f} GB, only "
                f"{free / 2**30:.2f} GB free under {self.dir}. Free space, or use fewer "
                f"bench.traj.methods, or bench.traj.save=false.")
        return need

    def write(self, name, tr):
        """Append a chunk: tr is (T, B, d), the next B seeds in order."""
        if name not in self._mm:
            path = os.path.join(self.dir, f"{name}.npy")
            self._mm[name] = [np.lib.format.open_memmap(path, mode="w+", dtype=self.dtype,
                                                        shape=self.shape), 0]
        mm, s = self._mm[name]
        B = int(tr.shape[1])
        mm[s:s + B] = tr.detach().to("cpu", torch.float32).numpy().transpose(1, 0, 2)
        self._mm[name][1] = s + B

    def close(self, meta, info):
        files = {}
        for name, (mm, s) in self._mm.items():
            if s != self.shape[0]:
                raise RuntimeError(f"trajectory {name}: wrote {s} of {self.shape[0]} seeds")
            mm.flush()
            files[name] = f"{name}.npy"
        self._mm.clear()
        np.savez(os.path.join(self.dir, "meta.npz"), **meta)
        info = dict(info, files=files, shape=list(self.shape), dtype=str(self.dtype),
                    layout="(n_hall, T, d); [:, k] is level T-1-k: [:, 0] start, [:, -1] final sample")
        json.dump(info, open(os.path.join(self.dir, "meta.json"), "w"), indent=2)
        return files


def load_traj(cell_dir):
    """One cached cell: (meta, info, {name: read-only memmap (n_hall, T, d)})."""
    meta = dict(np.load(os.path.join(cell_dir, "meta.npz")))
    info = json.load(open(os.path.join(cell_dir, "meta.json")))
    return meta, info, {m: np.load(os.path.join(cell_dir, f), mmap_mode="r") for m, f in info["files"].items()}


# ------------------------------------------------------------------ one cell
def bench_one(cfg, d, K, device, knobs=None, save=None):
    """One cell. knobs: a finished cell's row, whose lam / rho are then used instead of tuning
    (stage hall_bench_traj). save: cache trajectories (default bench.traj.save)."""
    b = cfg.bench
    got = load_cell(cfg, b, d, K, device, need_ddim=False)
    if got is None:
        print(f"[hall_bench] d={d} K={K}: no {ckpt_process(cfg, b)} checkpoint for seed "
              f"{ckpt_seed(cfg, b)} ({variant_of(cfg)}), skipped")
        return None
    model, ck, proc, means_t, R99, path = got
    S = make_sampler(proc, model, "learned")
    S_true = make_sampler(proc, model, "true")
    t0 = time.time()
    C = int(b.chunk)                                         # plain sampling
    Cg = int(b.get("grad_chunk", 64))                        # anything with autograd

    Z, L, how = ground_truth(cfg, d, K, S, S_true, means_t, R99, device, proc)
    hall = L < 0
    zh = Z[hall]
    N, n_h = int(Z.shape[0]), int(hall.sum())
    print(f"[hall_bench] {S.kind} d={d:>3} K={K:>2} T={S.T}: {n_h}/{N} ground-truth hallucinations "
          f"({100*n_h/max(N,1):.2f}%)  [{how}]", flush=True)
    row = {"d": d, "K": K, "T": int(S.T), "N": N, "gt": how, "n_hall": n_h, "sampler": S.kind}
    if n_h == 0:
        return row
    if n_h / max(N, 1) > float(b.get("max_hall_rate", 0.5)):
        print(f"[hall_bench]   {100*n_h/N:.1f}% hallucinating is above bench.max_hall_rate="
              f"{float(b.get('max_hall_rate', 0.5)):g}: the model or its labelling is broken here, cell skipped")
        row["skipped"] = "hall rate too high"
        return None

    tc_cfg = b.get("traj", None) or {}
    save = bool(tc_cfg.get("save", False)) if save is None else bool(save)
    tc = TrajCache(cfg, d, K, n_h, S.T) if save else None
    wants = lambda fam: tc is not None and tc.wants(fam)     # noqa: E731
    run_ours = bool(b.get("ours", True)) and (knobs is None or wants("ours"))
    run_iq = bool(b.get("iq", True)) and (knobs is None or wants("iq"))
    run_rods = bool(b.get("rods", True)) and (knobs is None or wants("rods"))
    windows = [float(w) for w in (b.get("iq_windows", None) or [b.iq_window])]
    if tc is not None:
        n_sets = (int(wants("original")) + len(b.eps_grid) * int(run_ours and wants("ours"))
                  + len(windows) * int(run_iq and wants("iq")) + 2 * int(run_rods and wants("rods")))
        need = tc.check_disk(n_sets)
        print(f"[hall_bench]   caching {n_sets} trajectory sets of {n_h}x{S.T}x{d} "
              f"({need / 2**30:.2f} GB) in {tc.dir}", flush=True)
    fates = {}

    def G(z, name=None):
        """Endpoints of the learned sampler; with a name, every state is cached as well."""
        if name is None:
            return torch.cat([S.G(c) for c in z.split(C)])
        ends = []
        for c in z.split(C):
            tr = S.traj(c)                                   # the same steps as S.G
            tc.write(name, tr)
            ends.append(tr[-1])
        return torch.cat(ends)

    def still(z_end):                                        # still hallucinating
        return int((core.label_fate(z_end, means_t, R99) < 0).sum())

    X_orig = G(zh, "original" if wants("original") else None)
    fates["original"] = core.label_fate(X_orig, means_t, R99)
    row["n_hall_plain"] = still(X_orig)                      # sanity: GT vs the learned sampler
    tgt = torch.cdist(X_orig, means_t).argmin(1)
    mu_t = means_t[tgt]

    # two kinds of hallucination
    is_interp, tube_d, _ = tube_class(X_orig, means_t, R99)
    cats = {"interp": is_interp, "invalid": ~is_interp}
    row["n_interp"], row["n_invalid"] = int(is_interp.sum()), int((~is_interp).sum())
    row["tube_d_med"] = float(tube_d[torch.isfinite(tube_d)].median()) if torch.isfinite(tube_d).any() else None
    ok_lab = lambda X: core.label_fate(X, means_t, R99) >= 0   # corrected = lands in a core

    def per_cat(fixed, tag):                                 # counts corrected in each category
        for c, m in cats.items():
            row[f"{tag}|{c}"] = int((fixed & m).sum())
        return int(fixed.sum())

    g = torch.Generator(device=device).manual_seed(int(b.seed) + 3)
    tune = torch.randperm(n_h, generator=g, device=device)[: min(int(b.tune_n), n_h)]

    # ours: the whole eps grid, so the strength can be read off rather than tuned away
    nrm = None
    if run_ours:
        nrm = torch.cat([pulled_normal(S, c, S.T - 1, m)[0] for c, m in zip(zh.split(Cg), mu_t.split(Cg))])
        best = None
        for e in [float(e) for e in b.eps_grid]:
            name = f"ours_eps{e:g}"
            end = G(zh - e * nrm, name if wants("ours") else None)
            fates[name] = core.label_fate(end, means_t, R99)
            fixed = fates[name] >= 0
            n_fix = per_cat(fixed, f"ours@{e:g}")
            if best is None or n_fix > best[1]:
                best = (e, n_fix)
        row["eps"], row["n_ours"] = best[0], n_h - best[1]    # best eps, and what it leaves

    # IQ at one or several windows (bench.iq_windows, e.g. [0.5, 0.4, 0.3, 0.2]; IQ acts on
    # every step with t = (i+1)/T <= window). The noise draws are taken either way so the RNG
    # stream, and every later choice, is the same whichever methods run.
    i0 = int(np.clip(round(float(b.iq_t0) * S.T - 1), 0, S.T - 1))
    MC = torch.randn(int(b.n_mc) // 2, d, generator=g, device=device)
    if run_iq:
        gradE = iq_energy(S, i0, torch.cat([MC, -MC]))

        for w in windows:
            def iq_end(lam, idx=slice(None), w=w, name=None):
                ends = []
                for c in zh[idx].split(Cg):
                    tr = sample_iq(S, gradE, c, lam, w)[0]
                    if name is not None:
                        tc.write(name, tr)
                    ends.append(tr[-1])
                return torch.cat(ends)
            if knobs is not None and f"lam@{w:g}" in knobs:
                lam = float(knobs[f"lam@{w:g}"])
            else:
                scan_l = {float(l): still(iq_end(float(l), tune)) for l in b.lam_grid}
                lam = min(scan_l, key=lambda l: (scan_l[l], l))
            name = f"iq_t{w:g}"
            end = iq_end(lam, name=name if wants("iq") else None)
            fates[name] = core.label_fate(end, means_t, R99)
            fixed = fates[name] >= 0
            row[f"n_iq@{w:g}"] = n_h - per_cat(fixed, f"iq@{w:g}")
            row[f"lam@{w:g}"] = lam
            if len(windows) == 1:
                row["n_iq"], row["lam"] = row[f"n_iq@{w:g}"], lam

    # RODS-SAS and RODS-CAS (bench.rods=false skips them)
    for kind in (("sas", "cas") if run_rods else ()):
        def rods_end(rho, idx=slice(None), name=None):
            rec = (lambda tr: tc.write(name, tr)) if name is not None else None
            return sample_rods(S, zh[idx], rho, kind, float(b.rods_thresh), b.rods_window, Cg, rec)
        if knobs is not None and f"rho_{kind}" in knobs:
            rho = float(knobs[f"rho_{kind}"])
        else:
            scan_r = {float(r): still(rods_end(float(r), tune)) for r in b.rho_grid}
            rho = min(scan_r, key=lambda r: (scan_r[r], r))
        name = f"rods_{kind}"
        end = rods_end(rho, name=name if wants("rods") else None)
        fates[name] = core.label_fate(end, means_t, R99)
        fixed = fates[name] >= 0
        row[f"n_{kind}"] = n_h - per_cat(fixed, kind)
        row[f"rho_{kind}"] = rho

    if tc is not None:
        levels = np.arange(S.T - 1, -1, -1)
        meta = {"seed_index": torch.nonzero(hall).flatten().cpu().numpy(), "z": zh.cpu().numpy(),
                "target_mode": tgt.cpu().numpy(), "interp": is_interp.cpu().numpy(),
                "tube_d": tube_d.cpu().numpy(), "means": means_t.cpu().numpy(), "R99": np.float64(R99),
                "levels": levels, "t": np.array([S.time(i) for i in levels]), "run_frac": (levels + 1) / S.T,
                "eps_grid": np.array([float(e) for e in b.eps_grid])}
        if nrm is not None:
            meta["normal"] = nrm.cpu().numpy()
        for name, f in fates.items():
            meta[f"fate_{name}"] = f.cpu().numpy()
        info = {"d": d, "K": K, "N": N, "n_hall": n_h, "gt": how, "checkpoint": path, "sampler": S.kind,
                "time": "meta.npz t: the sampler's own time per row (ddim: (i+1)/T, 1 = noise; flow: "
                        "0 = noise, 1 = data); run_frac: (i+1)/T for both",
                "fates": "meta.npz fate_<name>: mode index of the final sample, -1 = still hallucinated",
                "ours": "ours_eps<e> starts at z - e * normal (meta.npz z, normal)",
                "knobs": {k: v for k, v in row.items() if k.startswith(("lam@", "rho_"))}}
        files = tc.close(meta, info)
        print(f"[hall_bench]   cached trajectories: {', '.join(files)}", flush=True)

    row["secs"] = round(time.time() - t0, 1)
    parts = []
    if "n_ours" in row:
        parts.append(f"ours {row['n_ours']} (eps={row['eps']:g})")
    for k in sorted([k for k in row if k.startswith("n_iq@")], key=lambda k: -float(k[5:])):
        w = k[5:]
        parts.append(f"IQ(t<={w}) {row[k]} (lam={row['lam@' + w]:g})")
    if "n_sas" in row:
        parts.append(f"SAS {row['n_sas']} (rho={row['rho_sas']:g})  CAS {row['n_cas']} (rho={row['rho_cas']:g})")
    print(f"[hall_bench]   {n_h} hallucinated ({row['n_interp']} interpolation, "
          f"{row['n_invalid']} invalid) -> " + "  ".join(parts) + f"   [{row['secs']}s]", flush=True)
    return row


KEYS = ["d", "K", "T", "N", "n_hall", "n_hall_plain", "n_ours", "n_iq", "n_sas", "n_cas",
        "eps", "lam", "rho_sas", "rho_cas", "gt", "secs"]


def _save(cfg, rows):
    os.makedirs(bench_dir(cfg), exist_ok=True)
    json.dump(rows, open(os.path.join(bench_dir(cfg), "cells.json"), "w"), indent=2)
    path = os.path.join(bench_dir(cfg), "summary.csv")
    extra = sorted({k for r in rows for k in r if k not in KEYS and k != "gt"})
    keys = [k for k in KEYS if k != "gt"] + extra + ["gt"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k) for k in keys])
    return path


def run(cfg):
    device = core.get_device(cfg.device)
    prev = os.path.join(bench_dir(cfg), "cells.json")
    rows = json.load(open(prev)) if os.path.exists(prev) else []   # resume a crashed run
    done = {(r["d"], r["K"]) for r in rows}
    for d in cfg.bench.d:
        for K in cfg.bench.K:
            if (int(d), int(K)) in done:
                print(f"[hall_bench] d={d} K={K}: already done, skipped")
                continue
            r = bench_one(cfg, int(d), int(K), device)
            if r and r.get("n_hall"):
                rows.append(r)
                _save(cfg, rows)                            # after every cell
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    path = _save(cfg, rows)
    print(f"[hall_bench] wrote {path}")
    return {"summary": path, "n_cells": len(rows)}


def traj(cfg):
    """Stage hall_bench_traj: cache the trajectories of a finished benchmark (its cells.json) for
    bench.traj.methods. Same seeds, the lam / rho each cell chose, no tuning; cells.json is not
    touched. Cells whose meta.json exists are skipped, so re-running resumes."""
    device = core.get_device(cfg.device)
    prev = os.path.join(bench_dir(cfg), "cells.json")
    if not os.path.exists(prev):
        raise FileNotFoundError(f"{prev}: run stage hall_bench first (same run_id / data.weighted)")
    cells = {(r["d"], r["K"]): r for r in json.load(open(prev))}
    for d in cfg.bench.d:
        for K in cfg.bench.K:
            r = cells.get((int(d), int(K)))
            if r is None or r.get("skipped") or not r.get("n_hall"):
                continue
            if os.path.exists(os.path.join(traj_dir(cfg, int(d), int(K)), "meta.json")):
                print(f"[hall_bench_traj] d={d} K={K}: already cached, skipped")
                continue
            new = bench_one(cfg, int(d), int(K), device, knobs=r, save=True)
            if new is None:
                continue
            diff = {k: (r[k], new[k]) for k in new if "|" in k and k in r and r[k] != new[k]}
            if new.get("n_hall") != r.get("n_hall"):
                diff["n_hall"] = (r.get("n_hall"), new.get("n_hall"))
            print(f"[hall_bench_traj] d={d} K={K}: " + ("counts match cells.json" if not diff else
                  f"differs from cells.json (old, new): {diff}"), flush=True)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ------------------------------------------------------------------ table (stage hall_bench_viz)
def _cat_table(rows, cat, out, note=""):
    """One table per category: how many of that category each setting CORRECTED."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eps = sorted({float(k.split("@")[1].split("|")[0]) for r in rows for k in r
                  if k.startswith("ours@") and k.endswith("|" + cat)})
    iqw = sorted({float(k.split("@")[1].split("|")[0]) for r in rows for k in r
                  if k.startswith("iq@") and k.endswith("|" + cat)}, reverse=True)
    cols = ([(f"ours@{e:g}|{cat}", rf"$\epsilon$={e:g}") for e in eps]
            + [(f"iq@{w:g}|{cat}", f"IQ t<={w:g}") for w in iqw]
            + [(k, n) for k, n in ((f"sas|{cat}", "RODS-SAS"), (f"cas|{cat}", "RODS-CAS"))
               if all(k in r for r in rows)])
    cols = [c for c in cols if all(c[0] in r for r in rows)]
    n_key = "n_interp" if cat == "interp" else "n_invalid"
    rows = [r for r in rows if r.get(n_key)]
    if not rows or not cols:
        return None
    nr = len(rows) + 1
    fig, ax = plt.subplots(figsize=(4.6 + 1.15 * len(cols), 0.42 * nr + 1.7))
    ax.axis("off")
    colx = [0.035, 0.10] + list(np.linspace(0.22, 0.985, 1 + len(cols)))
    unit = 0.93 / (nr + 1.7)
    top = 0.975
    rowy = lambda i: top - unit * (1.9 + i)
    L = lambda y, lw, col="black": ax.plot([0.01, 0.995], [y, y], color=col, lw=lw,
                                           transform=ax.transAxes, clip_on=False)
    Tx = lambda x, y, t, **k: ax.text(x, y, t, transform=ax.transAxes, va="center", **k)
    L(top + unit * 0.2, 1.3)
    Tx((colx[2] + colx[1 + len(eps)]) / 2, top - unit * 0.4, "ours: step along $-n$", ha="center",
       fontsize=10, fontweight="bold")
    L(top - unit * 0.72, 0.6, "0.6")
    for j, h in enumerate(["$d$", "$K$", "in class"] + [c[1] for c in cols]):
        Tx(colx[j], top - unit * 1.25, h, ha="right" if j > 1 else "center", fontsize=10,
           fontweight="bold")
    L(top - unit * 1.62, 0.8)
    prev, tot = None, {c[0]: 0 for c in cols}
    tot_n = 0
    for i, r in enumerate(rows):
        y = rowy(i)
        if prev is not None and r["d"] != prev:
            L(y + unit * 0.5, 0.6, "0.75")
        prev = r["d"]
        Tx(colx[0], y, str(r["d"]), ha="center", fontsize=10)
        Tx(colx[1], y, str(r["K"]), ha="center", fontsize=10)
        Tx(colx[2], y, f"{r[n_key]:,}", ha="right", fontsize=10, color="0.4")
        tot_n += r[n_key]
        best = max(r[c[0]] for c in cols)
        for j, (k, _) in enumerate(cols):
            tot[k] += r[k]
            Tx(colx[3 + j], y, f"{r[k]:,}", ha="right", fontsize=10,
               fontweight="bold" if r[k] == best else "normal")
    y = rowy(len(rows))
    L(y + unit * 0.5, 0.8)
    Tx(colx[1], y, "total", ha="center", fontsize=10)
    Tx(colx[2], y, f"{tot_n:,}", ha="right", fontsize=10, color="0.4")
    bt = max(tot.values())
    for j, (k, _) in enumerate(cols):
        Tx(colx[3 + j], y, f"{tot[k]:,}", ha="right", fontsize=10,
           fontweight="bold" if tot[k] == bt else "normal")
    L(y - unit * 0.55, 1.3)
    name = "mode interpolations (inside the $R_{99}$ tube joining two modes)" if cat == "interp" \
        else "invalid samples (outside every core and every tube)"
    Tx(0.01, y - unit * 1.45, f"Hallucinations CORRECTED, of the {name}.", ha="left",
       fontsize=8.8, style="italic", color="0.3")
    Tx(0.01, y - unit * 2.1, "'in class' is how many of that kind there were; most per row in bold. "
       + note, ha="left", fontsize=8.8, style="italic", color="0.3")
    path = os.path.join(out, f"table_{cat}.png")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    o = [r"\begin{table}[t]", r"\centering",
         r"\caption{Hallucinations corrected, of the " + name.replace("$R_{99}$", "$R_{99}$") +
         r". 'in class' is how many of that kind there were; most per row in bold.}",
         r"\label{tab:hall-" + cat + "}",
         r"\begin{tabular}{rr r " + "r" * len(cols) + "}", r"\toprule",
         r"$d$ & $K$ & in class & " + " & ".join(c[1].replace("<=", r"$\le$") for c in cols) + r" \\",
         r"\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["d"] != prev:
            o.append(r"\midrule")
        prev = r["d"]
        best = max(r[c[0]] for c in cols)
        cells = [(r"\textbf{%d}" % r[k]) if r[k] == best else str(r[k]) for k, _ in cols]
        o.append(f"{r['d']} & {r['K']} & {r[n_key]} & " + " & ".join(cells) + r" \\")
    o.append(r"\midrule")
    cells = [(r"\textbf{%d}" % tot[k]) if tot[k] == bt else str(tot[k]) for k, _ in cols]
    o.append(r"\multicolumn{2}{r}{total} & " + f"{tot_n} & " + " & ".join(cells) + r" \\")
    o += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, f"table_{cat}.tex"), "w").write("\n".join(o) + "\n")
    return path


def viz(cfg):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.load(open(os.path.join(bench_dir(cfg), "cells.json")))
    if not rows:
        print("[hall_bench_viz] nothing to draw")
        return {}
    rows.sort(key=lambda r: (r["d"], r["K"]))
    out = fig_dir(cfg)
    os.makedirs(out, exist_ok=True)
    iqk = sorted({k for r in rows for k in r if k.startswith("n_iq@")}, key=lambda k: -float(k[5:]))
    cand = ([("n_ours", "Ours")] + [(k, f"IQ t<={k[5:]}") for k in iqk]
            + ([] if iqk else [("n_iq", "IQ")]) + [("n_sas", "RODS-SAS"), ("n_cas", "RODS-CAS")])
    meth = [m for m in cand if all(m[0] in r for r in rows)]
    N = rows[0]["N"]

    nr = len(rows) + 1
    fig, ax = plt.subplots(figsize=(6.5 + 1.45 * len(meth), 0.42 * nr + 1.5))
    ax.axis("off")
    colx = [0.04, 0.11] + list(np.linspace(0.30, 0.97, 1 + len(meth)))
    unit = 0.93 / (nr + 1.4)
    top = 0.975
    rowy = lambda i: top - unit * (1.6 + i)
    Lh = lambda y, lw, col="black": ax.plot([0.01, 0.99], [y, y], color=col, lw=lw,
                                            transform=ax.transAxes, clip_on=False)
    Tx = lambda x, y, t, **k: ax.text(x, y, t, transform=ax.transAxes, va="center", **k)
    Lh(top + unit * 0.2, 1.3)
    for j, h in enumerate(["$d$", "$K$", "original"] + [m[1] for m in meth]):
        Tx(colx[j], top - unit * 0.55, h, ha="right" if j > 1 else "center", fontsize=11,
           fontweight="bold")
    Lh(top - unit * 1.05, 0.8)
    prev = None
    tot = {k: 0 for k in ["n_hall"] + [m[0] for m in meth]}
    for i, r in enumerate(rows):
        y = rowy(i)
        if prev is not None and r["d"] != prev:
            Lh(y + unit * 0.5, 0.6, "0.75")
        prev = r["d"]
        Tx(colx[0], y, str(r["d"]), ha="center", fontsize=10.5)
        Tx(colx[1], y, str(r["K"]), ha="center", fontsize=10.5)
        Tx(colx[2], y, f"{r['n_hall']:,}", ha="right", fontsize=10.5, color="0.35")
        best = min(r[m[0]] for m in meth)
        for j, (k, _) in enumerate(meth):
            Tx(colx[3 + j], y, f"{r[k]:,}", ha="right", fontsize=10.5,
               fontweight="bold" if r[k] == best else "normal")
        for k in tot:
            tot[k] += r[k]
    y = rowy(len(rows))
    Lh(y + unit * 0.5, 0.8)
    Tx(colx[1], y, "total", ha="center", fontsize=10.5)
    Tx(colx[2], y, f"{tot['n_hall']:,}", ha="right", fontsize=10.5, color="0.35")
    bt = min(tot[m[0]] for m in meth)
    for j, (k, _) in enumerate(meth):
        Tx(colx[3 + j], y, f"{tot[k]:,}", ha="right", fontsize=10.5,
           fontweight="bold" if tot[k] == bt else "normal")
    Lh(y - unit * 0.55, 1.3)
    Tx(0.01, y - unit * 1.4, f"Hallucinated samples out of N = {N:,} seeds per cell: the ground-truth "
       f"count, and how many remain after each repair (fewest per row in bold).", ha="left",
       fontsize=8.8, style="italic", color="0.3")
    Tx(0.01, y - unit * 2.05, "IQ t<=w: IQ on every step with t = (i+1)/T <= w; each method's step size "
       "/ strength chosen on a subset of the hallucinated seeds.", ha="left",
       fontsize=8.8, style="italic", color="0.3")
    fig.savefig(os.path.join(out, "table_hall_bench.png"), dpi=220, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    o = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Hallucinated samples out of $N={N:,}$ seeds per cell: the ground-truth count "
         r"and the number remaining after each repair, applied to the ground-truth hallucinations "
         r"only. IQ $t\le w$: IQ on every sampling step with $t=(i+1)/T \le w$. Fewest per row in bold.}",
         r"\label{tab:hall-bench}", r"\begin{tabular}{rr r " + "r" * len(meth) + "}", r"\toprule",
         r"$d$ & $K$ & original & " + " & ".join(m[1] for m in meth) + r" \\", r"\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["d"] != prev:
            o.append(r"\midrule")
        prev = r["d"]
        best = min(r[m[0]] for m in meth)
        cells = [(r"\textbf{%d}" % r[k]) if r[k] == best else str(r[k]) for k, _ in meth]
        o.append(f"{r['d']} & {r['K']} & {r['n_hall']} & " + " & ".join(cells) + r" \\")
    o.append(r"\midrule")
    cells = [(r"\textbf{%d}" % tot[k]) if tot[k] == bt else str(tot[k]) for k, _ in meth]
    o.append(r"\multicolumn{2}{r}{total} & " + f"{tot['n_hall']} & " + " & ".join(cells) + r" \\")
    o += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "table_hall_bench.tex"), "w").write("\n".join(o) + "\n")
    made = [os.path.join(out, "table_hall_bench")]
    for cat in ("interp", "invalid"):
        p_ = _cat_table(rows, cat, out, f"IQ / RODS at their tuned settings (lambda, rho).")
        if p_:
            made.append(p_[:-4])
            print(f"[hall_bench_viz] wrote {p_}")
    print(f"[hall_bench_viz] wrote {out}/table_hall_bench.png/.tex")
    return {"figures": made}


# ==================================================================================== #
#  stage: hall_bench_anim — replay benchmark seeds with every step recorded (d = 2)     #
# ==================================================================================== #
"""Reads the finished benchmark (output/<run_id>/ddim/<variant>/hall_bench/cells.json), rebuilds
each cell's hallucinated seeds (cells with d in bench.anim_d, default [2]) exactly as the benchmark did (same ground truth, same RNG
stream), takes the first bench.anim_n of EACH class (mode interpolation / invalid; no cherry-picking),
and re-runs every method
with the settings the benchmark chose (eps, lam per IQ window, rho), recording full paths.
d = 2 is drawn directly; d > 2 in the (a, r) plane of each seed's two modes, where distances to
both centres are exact. Writes one animation (gif, + mp4 with ffmpeg) and one still per seed to
output/<run_id>/ddim/<variant>/hall_bench/figures/. Use the benchmark's run_id and data.weighted.

    python code/baselines/main.py 'stages=[hall_bench_anim]' run_id=<benchmark run_id>
"""


def sample_rods_traj(S, z, rho, kind, thresh, window):
    """sample_rods, recording every state; returns (T, B, d)."""
    T = S.T
    lo, hi = float(window[0]), float(window[1])
    x, xs = z, [z]
    for k, i in enumerate(range(T - 1, 0, -1)):
        frac = k / max(1, T - 2)
        if rho <= 0 or not (lo <= frac <= hi):
            with torch.no_grad():
                x = S.step(x, i)
        else:
            gn = grad_score_norm(S, x, i)
            u = gn / gn.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if thresh > 0:
                on = (grad_score_norm(S, x + rho * u, i) - gn).norm(dim=1) >= thresh
            else:
                on = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            with torch.no_grad():
                if kind == "sas":
                    s_ = S.score(x, i)
                    delta = -rho * s_ / s_.norm(dim=1, keepdim=True).clamp_min(1e-12)
                else:
                    delta = rho * u
                e = torch.where(on[:, None], S.pred(x + delta, i), S.pred(x, i))
                x = S.update(x, i, e)
        xs.append(x.detach())
    return torch.stack(xs)


def _anim_multi(paths, t, MU, R99, zh, zo, nn, eps, title, path, fps=25, hold=20, intro=20,
                max_frames=120, dpi=110):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    from matplotlib.colors import ListedColormap
    import shutil
    K = MU.shape[0]
    pale = ["#fde0dd", "#deebf7", "#e5f5e0", "#fff7bc", "#efedf5", "#fee6ce", "#e0f3f8",
            "#f2f0f7", "#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc",
            "#e5d8bd", "#fddaec"]
    cols = {"original": C_OG, "ours": C_US, "RODS-SAS": "#9467bd", "RODS-CAS": "#2ca02c"}
    iq_cols = ["#1f77b4", "#4a98c9", "#7ab6d9", "#a6cde3"]
    names = list(paths)
    for j, nm in enumerate([n for n in names if n.startswith("IQ")]):
        cols[nm] = iq_cols[j % len(iq_cols)]
    n = len(t)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    pts = np.concatenate([p for p in paths.values()] + [MU])
    lo_, hi_ = pts.min(0), pts.max(0)
    c, w = 0.5 * (lo_ + hi_), 0.62 * max(hi_ - lo_) + 2 * R99
    gx, gy = np.linspace(c[0] - w, c[0] + w, 300), np.linspace(c[1] - w, c[1] + w, 300)
    GX, GY = np.meshgrid(gx, gy)
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    near = np.argmin(((P[:, None, :] - MU[None]) ** 2).sum(-1), 1).reshape(GX.shape)
    ax[0].pcolormesh(GX, GY, near, cmap=ListedColormap(pale[:K]), shading="auto",
                     vmin=0, vmax=K - 1, alpha=0.65)
    ax[0].contour(GX, GY, near, levels=np.arange(K) + 0.5, colors="0.55", linestyles="--",
                  linewidths=0.9)
    th = np.linspace(0, 2 * np.pi, 240)
    for k in range(K):
        ax[0].plot(MU[k, 0] + R99 * np.cos(th), MU[k, 1] + R99 * np.sin(th), color="0.35", lw=1.2)
        ax[0].plot(*MU[k], "k*", ms=12)
    ax[0].plot(*paths["original"][0], "X", color="k", ms=11, zorder=5)
    ax[0].set_xlim(c[0] - w, c[0] + w); ax[0].set_ylim(c[1] - w, c[1] + w)
    ax[0].set_aspect("equal"); ax[0].set_xlabel("$x_1$"); ax[0].set_ylabel("$x_2$")
    wi = max(2.2 * eps, 1e-6)                                # inset: the pullback step
    ins = ax[0].inset_axes([1.0 - 0.30, 0.0, 0.30, 0.30])   # bottom-right corner, small
    ins.set_xlim(zh[0] - wi, zh[0] + wi); ins.set_ylim(zh[1] - wi, zh[1] + wi)
    ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("white")
    ins.annotate("", xy=zh - 0.8 * wi * nn, xytext=zh, arrowprops=dict(arrowstyle="->", color="k", lw=1.6))
    ins.text(0.05, 0.05, rf"$-n$,  $\epsilon$={eps:.3g}", transform=ins.transAxes, fontsize=8)
    ins.plot(*zh, "X", color="k", ms=9); ins.plot(*zo, "o", color=C_US, ms=7, mec="k")
    ins.set_title("pullback step at $x_T$", fontsize=8)
    order = [k for k in names if k != "original"] + ["original"]   # original on top, dashed
    lines = {k: ax[0].plot([], [], color=cols[k],
                           lw=1.3 if k == "original" else (2.2 if k == "ours" else 1.8),
                           ls=(0, (4, 3)) if k == "original" else "-", label=k,
                           zorder=6 if k == "original" else 3)[0] for k in order}
    dots = {k: ax[0].plot([], [], "o", color=cols[k], ms=8, mec="k",
                          zorder=7 if k == "original" else 4)[0] for k in order}
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
                 framealpha=0.92, facecolor="white")
    D = {k: np.linalg.norm(paths[k] - paths["original"], axis=1) for k in names if k != "original"}
    for k in D:
        ax[1].plot(t, D[k], color=cols[k], lw=1, alpha=0.18)
    dl = {k: ax[1].plot([], [], color=cols[k], lw=2, label=k)[0] for k in D}
    ax[1].axhline(R99, color="0.5", lw=1, ls=":", label=r"$R_{99}$")
    now = ax[1].axvline(t[0], color="0.3", lw=1)
    allD = np.concatenate([v[1:] for v in D.values()])
    ax[1].set_yscale("log"); ax[1].set_ylim(max(1e-4, allD[allD > 0].min()) * 0.5, allD.max() * 2)
    ax[1].set_xlim(t[0], t[-1]); ax[1].set_xlabel("t  (sampling runs 1 → 0)")
    ax[1].set_ylabel(r"$\|x_t-x_t^{\mathrm{orig}}\|$")
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
                 framealpha=0.92, facecolor="white")
    sup = fig.suptitle("")
    idx = (np.unique(np.linspace(0, n - 1, max_frames).astype(int)) if n > max_frames else np.arange(n))

    def frame(f):
        if f < intro:
            i = 0
            sup.set_text(f"{title}:  pullback step, " + rf"$\epsilon$ = {eps:.3g}")
        else:
            i = int(idx[min(f - intro, len(idx) - 1)])
            sup.set_text(f"{title}:  reverse step {i}/{n-1}   (t = {t[i]:.3f})")
        for k in names:
            lines[k].set_data(paths[k][: i + 1, 0], paths[k][: i + 1, 1])
            dots[k].set_data([paths[k][i, 0]], [paths[k][i, 1]])
        for k in D:
            dl[k].set_data(t[: i + 1], D[k][: i + 1])
        now.set_xdata([t[i], t[i]])
        return list(lines.values()) + list(dots.values()) + list(dl.values()) + [now, sup]

    fig.tight_layout()
    frame(intro + len(idx) - 1)                              # still of the final state
    fig.savefig(f"{path}.png", dpi=170)
    an = FuncAnimation(fig, frame, frames=intro + len(idx) + hold, interval=1000 / fps, blit=False)
    if shutil.which("ffmpeg"):
        an.save(f"{path}.mp4", writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
    an.save(f"{path}.gif", writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def _anim_meridian(paths_x0, paths_x, t, mu1, mu2, R99, eps, title, path, fps=25, hold=20,
                   max_frames=120, dpi=110):
    """d > 2: plot the predicted clean sample x0_hat(x_t) in (a, r) coordinates, where a is the
    position along the axis mu1 -> mu2 and r the distance from that axis. Distances to BOTH mode
    centres are exact in these coordinates, so both cores are exact half-disks of radius R99
    and the boundary between the two modes is exact."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    import shutil
    e = (mu2 - mu1) / np.linalg.norm(mu2 - mu1)
    L = float(np.linalg.norm(mu2 - mu1))

    def ar(X):                                              # (T, d) -> (T, 2)
        Y = X - mu1
        a = Y @ e
        r = np.linalg.norm(Y - a[:, None] * e[None], axis=1)
        return np.stack([a, r], 1)

    P = {k: ar(v) for k, v in paths_x0.items()}
    cols = {"original": C_OG, "ours": C_US, "RODS-SAS": "#9467bd", "RODS-CAS": "#2ca02c"}
    iq_cols = ["#1f77b4", "#4a98c9", "#7ab6d9", "#a6cde3"]
    names = list(P)
    for j, nm in enumerate([n for n in names if n.startswith("IQ")]):
        cols[nm] = iq_cols[j % len(iq_cols)]
    n = len(t)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    xlo, xhi = -2.5 * R99, L + 2.5 * R99
    yhi = max(2.5 * R99, 0.6 * L)
    ax[0].axvspan(xlo, L / 2, color="#deebf7", alpha=0.65, lw=0)
    ax[0].axvspan(L / 2, xhi, color="#fde0dd", alpha=0.65, lw=0)
    ax[0].axvline(L / 2, color="0.55", ls="--", lw=0.9)
    th = np.linspace(0, np.pi, 200)
    for cx, lab in ((0.0, r"$\mu_{target}$"), (L, r"$\mu_{other}$")):
        ax[0].plot(cx + R99 * np.cos(th), R99 * np.sin(th), color="0.35", lw=1.2)
        ax[0].plot(cx, 0, "k*", ms=12, clip_on=False)
        ax[0].annotate(lab, (cx, 0), textcoords="offset points", xytext=(6, 6), fontsize=10)
    ax[0].set_xlim(xlo, xhi); ax[0].set_ylim(0, yhi); ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$a$: position along the axis $\mu_{target} \to \mu_{other}$")
    ax[0].set_ylabel(r"$r$: distance from that axis")
    ax[0].set_title(r"predicted clean sample $\hat x_0(x_t)$; distances to both modes exact", fontsize=10)
    order = [k for k in names if k != "original"] + ["original"]   # original on top, dashed
    lines = {k: ax[0].plot([], [], color=cols[k],
                           lw=1.3 if k == "original" else (2.2 if k == "ours" else 1.8),
                           ls=(0, (4, 3)) if k == "original" else "-", label=k,
                           zorder=6 if k == "original" else 3)[0] for k in order}
    dots = {k: ax[0].plot([], [], "o", color=cols[k], ms=8, mec="k",
                          zorder=7 if k == "original" else 4)[0] for k in order}
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
                 framealpha=0.92, facecolor="white")
    D = {k: np.linalg.norm(paths_x[k] - paths_x["original"], axis=1) for k in names if k != "original"}
    for k in D:
        ax[1].plot(t, D[k], color=cols[k], lw=1, alpha=0.18)
    dl = {k: ax[1].plot([], [], color=cols[k], lw=2, label=k)[0] for k in D}
    ax[1].axhline(R99, color="0.5", lw=1, ls=":", label=r"$R_{99}$")
    now = ax[1].axvline(t[0], color="0.3", lw=1)
    allD = np.concatenate([v[1:] for v in D.values()])
    ax[1].set_yscale("log"); ax[1].set_ylim(max(1e-4, allD[allD > 0].min()) * 0.5, allD.max() * 2)
    ax[1].set_xlim(t[0], t[-1]); ax[1].set_xlabel("t  (sampling runs 1 → 0)")
    ax[1].set_ylabel(r"$\|x_t-x_t^{\mathrm{orig}}\|$ (full $d$-dim distance)")
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
                 framealpha=0.92, facecolor="white")
    sup = fig.suptitle("")
    idx = (np.unique(np.linspace(0, n - 1, max_frames).astype(int)) if n > max_frames else np.arange(n))

    def frame(f):
        i = int(idx[min(f, len(idx) - 1)])
        sup.set_text(f"{title}:  reverse step {i}/{n-1}   (t = {t[i]:.3f})   " + rf"$\epsilon$ = {eps:.3g}")
        for k in names:
            lines[k].set_data(P[k][: i + 1, 0], P[k][: i + 1, 1])
            dots[k].set_data([P[k][i, 0]], [P[k][i, 1]])
        for k in D:
            dl[k].set_data(t[: i + 1], D[k][: i + 1])
        now.set_xdata([t[i], t[i]])
        return list(lines.values()) + list(dots.values()) + list(dl.values()) + [now, sup]

    fig.tight_layout()
    frame(len(idx) - 1)
    fig.savefig(f"{path}.png", dpi=170)
    an = FuncAnimation(fig, frame, frames=len(idx) + hold, interval=1000 / fps, blit=False)
    if shutil.which("ffmpeg"):
        an.save(f"{path}.mp4", writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
    an.save(f"{path}.gif", writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def anim(cfg):
    b = cfg.bench
    device = core.get_device(cfg.device)
    cells = json.load(open(os.path.join(bench_dir(cfg), "cells.json")))
    out = fig_dir(cfg)
    os.makedirs(out, exist_ok=True)
    made = []
    want = [int(x) for x in (b.get("anim_d", None) or [2])]
    for r in [r for r in cells if int(r["d"]) in want]:
        d, K = int(r["d"]), int(r["K"])
        got = load_cell(cfg, b, d, K, device, need_ddim=False)
        if got is None:
            print(f"[hall_bench_anim] d={d} K={K}: checkpoint missing, skipped")
            continue
        model, ck, proc, means_t, R99, path = got
        S = make_sampler(proc, model, "learned")
        S_true = make_sampler(proc, model, "true")
        Z, L, how = ground_truth(cfg, d, K, S, S_true, means_t, R99, device, proc)
        zh_all = Z[L < 0]
        if zh_all.shape[0] != int(r["n_hall"]):
            print(f"[hall_bench_anim] d={d} K={K}: {zh_all.shape[0]} hallucinated seeds here vs "
                  f"{r['n_hall']} in the benchmark; check the run settings (variant, n)")
        g = torch.Generator(device=device).manual_seed(int(b.seed) + 3)   # same RNG stream
        torch.randperm(zh_all.shape[0], generator=g, device=device)
        MC = torch.randn(int(b.n_mc) // 2, d, generator=g, device=device)
        # take the first few of EACH class, so both kinds of hallucination get animated
        with torch.no_grad():
            X_all = torch.cat([S.G(c) for c in zh_all.split(int(b.chunk))])
        is_interp, _, _ = tube_class(X_all, means_t, R99)
        k_each = int(b.get("anim_n", 3))
        pick, klass = [], []
        for name, mask in (("interp", is_interp), ("invalid", ~is_interp)):
            idx = torch.nonzero(mask).flatten()[:k_each]
            pick.append(idx)
            klass += [name] * int(idx.numel())
        pick = torch.cat(pick)
        if pick.numel() == 0:
            continue
        zh = zh_all[pick]
        with torch.no_grad():
            tgt = torch.cdist(S.G(zh), means_t).argmin(1)
        mu_t = means_t[tgt]
        paths = {"original": S.traj(zh)}
        eps = float(b.get("anim_eps", None) or r.get("eps", 0.0) or 0.0)
        nrm = pulled_normal(S, zh, S.T - 1, mu_t)[0]
        z_ours = zh - eps * nrm
        if "n_ours" in r:
            paths["ours"] = S.traj(z_ours)
        i0 = int(np.clip(round(float(b.iq_t0) * S.T - 1), 0, S.T - 1))
        gradE = iq_energy(S, i0, torch.cat([MC, -MC]))
        for k in sorted([k for k in r if k.startswith("n_iq@")], key=lambda k: -float(k[5:])):
            w = k[5:]
            paths[f"IQ t<={w}"] = sample_iq(S, gradE, zh, float(r[f"lam@{w}"]), float(w))[0]
        for kind in ("sas", "cas"):
            if f"n_{kind}" in r:
                paths[f"RODS-{kind.upper()}"] = sample_rods_traj(
                    S, zh, float(r[f"rho_{kind}"]), kind, float(b.rods_thresh), b.rods_window)
        t = np.array([S.t_of(level(k, S.T)) for k in range(S.T)])
        cpu = lambda x: x.detach().cpu().numpy()
        MU = cpu(means_t)
        frames = int(b.get("anim_frames", None) or cfg.pullback.get("anim_frames", 120))
        if d > 2:                                           # predicted clean samples along each path
            with torch.no_grad():
                paths_x0 = {k: torch.stack([S.tweedie(v[kk], level(kk, S.T)) for kk in range(S.T)])
                            for k, v in paths.items()}
            with torch.no_grad():                           # the other end of each seed's channel
                two = torch.cdist(paths["original"][-1], means_t).topk(2, dim=1, largest=False).indices
        for j in range(zh.shape[0]):
            stem = os.path.join(out, f"anim_d{d}_K{K}_{klass[j]}{j}")
            ttl = f"d={d}, K={K}, {'mode interpolation' if klass[j] == 'interp' else 'invalid sample'}"
            if d == 2:
                P = {k: cpu(v[:, j]) for k, v in paths.items()}
                _anim_multi(P, t, MU, R99, cpu(zh[j]), cpu(z_ours[j]), cpu(nrm[j]), eps,
                            ttl, stem, max_frames=frames)
            else:
                t_i = int(tgt[j]); o_i = int(two[j, 1] if int(two[j, 0]) == t_i else two[j, 0])
                _anim_meridian({k: cpu(v[:, j]) for k, v in paths_x0.items()},
                               {k: cpu(v[:, j]) for k, v in paths.items()}, t,
                               MU[t_i], MU[o_i], R99, eps, ttl, stem, max_frames=frames)
            made.append(stem)
            print(f"[hall_bench_anim] wrote {stem}.gif", flush=True)
    return {"figures": made}