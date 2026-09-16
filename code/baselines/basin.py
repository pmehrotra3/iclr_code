"""baselines/basin.py — stage `basin`: does each seed reach the basin it should?

Why this stage exists
---------------------
`stages/evaluate` asks how PREDICTABLE a sampler's fates are from the seed: it
picks one ground-truth sampler and scores classifiers against it. That is the
right question for the paper's main claim, but it is not the question Abhinav
asked about the baselines, which is whether an intervention sends each seed to
the basin it *should* reach.

So this stage fixes one seed set and one reference, and compares samplers on it:

    reference   the analytic-field sampler (exact GMM score).  The basin a seed
                belongs to is defined by where the exact reverse map takes it --
                this is the paper's notion, not a distance threshold.
    candidates  the learnt sampler (`ddim`) and each intervention
                (`rods_cas`, `rods_sas`, `iq`).

Every metric is agreement between two fate vectors on the SAME seeds, so no
k*sigma / chi^2 / 3-sigma threshold appears anywhere. That also removes the
reason our numbers, Pranav's R99 numbers and the LID paper's 3-sigma numbers
have not been comparable.

Reported per (d, K), per sampler:

    basin_acc     fraction of ALL seeds sent to the basin the reference gives
                  (the headline: "does it go where it should")
    mode_acc      same, restricted to seeds the reference sends to a real mode
    misassigned   reference says mode i, sampler says mode j != i.  A wrong
                  class, NOT a hallucination -- the k*sigma rule cannot see this
                  at all, which is the main thing it was missing.
    hallucinated  reference says mode i, sampler says -1 (fell in the intermodal
                  intersection)
    spurious      reference says -1, sampler says a mode
    hall_rate     share of -1 under this sampler, for continuity with the old tables
    d_hall        hall_rate minus the reference's, signed

Against the learnt baseline, so an intervention is judged on what it changed:

    preserved     of seeds the BASELINE placed in a real basin, the fraction this
                  sampler still places in the SAME basin
    broke         baseline placed it, this sampler hallucinates it
    rescued       baseline hallucinated it, this sampler places it
    moved         baseline placed it in basin i, this sampler in basin j != i

Writes output/<process>/basin[_<tag>].{json,csv} and a per-(d,K) console table.
Run it under any one process; it loops over `basin.samplers` itself, so a single
invocation covers every method.
"""
from __future__ import annotations

import os
import json

import torch
from omegaconf import OmegaConf

from common import checkpoint, utils
from common.process import make_process

DEFAULT_SAMPLERS = ("ddim", "rods_cas", "rods_sas", "iq")


@torch.no_grad()
def _label(proc, model, X, R99, chunk=200000):
    out = []
    for j in range(0, X.shape[0], chunk):
        out.append(proc.label(model, X[j:j + chunk], R99))
    return torch.cat(out)


def _agree(ref, got):
    """Agreement of `got` against the reference fates `ref`. Fractions of ALL seeds."""
    n = ref.numel()
    ref_mode = ref >= 0
    return dict(
        basin_acc=float((got == ref).float().mean()),
        mode_acc=(float((got[ref_mode] == ref[ref_mode]).float().mean())
                  if bool(ref_mode.any()) else float("nan")),
        misassigned=float(((got >= 0) & ref_mode & (got != ref)).sum()) / n,
        hallucinated=float(((got == -1) & ref_mode).sum()) / n,
        spurious=float(((got >= 0) & (ref == -1)).sum()) / n,
        hall_rate=float((got == -1).float().mean()),
    )


def _vs_baseline(base, got):
    """What this sampler changed relative to the learnt baseline."""
    ok = base >= 0
    bad = ~ok
    f = lambda m: float(m.sum()) / max(int(ok.sum()), 1)          # noqa: E731
    return dict(
        preserved=(float((got[ok] == base[ok]).float().mean())
                   if bool(ok.any()) else float("nan")),
        broke=f((got == -1) & ok),
        moved=f((got >= 0) & ok & (got != base)),
        rescued=(float((got[bad] >= 0).float().mean())
                 if bool(bad.any()) else float("nan")),
    )


def _one(cfg, d, K, device, samplers):
    """All samplers on one (d, K) cell, sharing seeds and a reference."""
    ck_root = cfg.paths.checkpoints
    T_train = int(cfg.sweep.T_train)

    # any available checkpoint defines the GMM; prefer the learnt baseline's
    src = None
    for nm in ("ddim",) + tuple(samplers):
        p = utils.ckpt_path(ck_root, nm, d, K, T_train)
        if os.path.exists(p):
            src = p
            break
    if src is None:
        return None
    model, ck = checkpoint.load(src, device)
    means_t, R99, variance = ck["means"], ck["R99"], ck["variance"]

    n = int(cfg.basin.get("n_eval", cfg.eval.n_eval))
    proc0 = make_process("ddim", means_t, variance, ck["T"], device, cfg)
    X = proc0.seeds(n, d, cfg.seed + 1)

    # reference: the analytic field at T_true. No model, no threshold on top.
    ref_proc = make_process("ddim", means_t, variance, int(cfg.sweep.T_true),
                            device, cfg)
    ref = _label(ref_proc, None, X, R99)

    row = {"d": d, "K": K, "n_eval": n,
           "ref_hall_rate": float((ref == -1).float().mean()),
           "samplers": {}}

    base_fate = None
    for nm in samplers:
        p = utils.ckpt_path(ck_root, nm, d, K, T_train)
        if not os.path.exists(p):
            print(f"[basin] d={d:>2} K={K:>2} {nm:>9}: no checkpoint, skipped")
            continue
        m, _ = checkpoint.load(p, device)
        proc = make_process(nm, means_t, variance, ck["T"], device, cfg)
        got = _label(proc, m, X, R99)
        if nm == "ddim":
            base_fate = got
        r = _agree(ref, got)
        r["d_hall"] = r["hall_rate"] - row["ref_hall_rate"]
        if base_fate is not None and nm != "ddim":
            r.update(_vs_baseline(base_fate, got))
        row["samplers"][nm] = r
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def _print(row):
    d, K = row["d"], row["K"]
    print(f"\n=== d={d}  K={K}  n={row['n_eval']}  "
          f"reference (analytic) hall={100*row['ref_hall_rate']:.2f}% ===")
    print(f"  {'sampler':<10} {'basin acc':>10} {'mode acc':>9} {'misasgn':>8} "
          f"{'halluc':>7} {'spur':>6} {'HR':>7} {'preserved':>10} {'broke':>7} "
          f"{'moved':>7} {'rescued':>8}")
    for nm, r in row["samplers"].items():
        g = lambda k: (f"{100*r[k]:.2f}" if k in r else "-")      # noqa: E731
        print(f"  {nm:<10} {100*r['basin_acc']:>9.2f}% {100*r['mode_acc']:>8.2f}% "
              f"{100*r['misassigned']:>7.2f}% {100*r['hallucinated']:>6.2f}% "
              f"{100*r['spurious']:>5.2f}% {100*r['hall_rate']:>6.2f}% "
              f"{g('preserved'):>10} {g('broke'):>7} {g('moved'):>7} "
              f"{g('rescued'):>8}")


def run(cfg):
    device = utils.get_device(cfg.device)
    samplers = tuple(cfg.basin.get("samplers", DEFAULT_SAMPLERS))
    out_dir = utils.process_dir(cfg.paths.output, cfg.process)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            r = _one(cfg, int(d), int(K), device, samplers)
            if r is None:
                print(f"[basin] d={d:>2} K={K:>2} -> no checkpoint, skipped")
                continue
            _print(r)
            rows.append(r)

    tag = cfg.eval.tag
    js = os.path.join(out_dir, f"basin{'_' + tag if tag else ''}.json")
    with open(js, "w") as f:
        json.dump({"reference": "analytic field (exact GMM score)",
                   "samplers": list(samplers),
                   "sweep": OmegaConf.to_container(cfg.sweep, resolve=True),
                   "results": rows}, f, indent=2)

    cols = ["basin_acc", "mode_acc", "misassigned", "hallucinated", "spurious",
            "hall_rate", "d_hall", "preserved", "broke", "moved", "rescued"]
    csv = js[:-5] + ".csv"
    with open(csv, "w") as f:
        f.write("d,K,n_eval,ref_hall_rate,sampler," + ",".join(cols) + "\n")
        for r in rows:
            pre = f"{r['d']},{r['K']},{r['n_eval']},{r['ref_hall_rate']:.5f}"
            for nm, s in r["samplers"].items():
                f.write(f"{pre},{nm}," + ",".join(
                    (f"{s[c]:.5f}" if c in s else "") for c in cols) + "\n")
    print(f"\n[basin] wrote {js} and {csv}")
    return {"json": js, "csv": csv, "n_cells": len(rows)}