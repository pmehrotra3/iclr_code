"""
evaluate.py — Stage 2. For the SELECTED sampler, compare the analytic-field atlas
(all-anchor Gaussian vote) and the analytic-responsibility predictor against the learned
model's ground-truth fate, sweeping over anchor counts. Results go to
output/<sampler>/results.{json,csv}. No plotting here.
"""
from __future__ import annotations
import os
import json
import time
import torch

import core
from processes.factory import make_process
from train import ckpt_path, sampler_dir


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device)
    a = ck["arch"]
    m = core.ScoreNet(ck["d"], a["h"], a["nb"], a["td"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    ck["means"] = ck["means"].to(device)
    return m, ck


def eval_one(cfg, d, K, device):
    sampler = cfg.sampler
    path = ckpt_path(cfg.paths.data, sampler, d, K)
    if not os.path.exists(path):
        return None
    model, ck = load_ckpt(path, device)
    means_t = ck["means"]
    T, R99, variance = ck["T"], ck["R99"], ck["variance"]

    proc = make_process(sampler, means_t, variance, T, device, cfg)

    # ground truth from the learned model
    X0 = proc.seeds(cfg.eval.n_eval, d, cfg.seed + 1)
    Xf = proc.sample(model, X0)
    gt = core.label_fate(Xf, means_t, R99)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    m_cls = gt >= 0
    hall_gt = float((gt == -1).float().mean())

    # analytic responsibility (no anchors) — uses the source-scale for ddim; for flow the
    # source is t=0 with unit-variance seeds, so responsibilities at the seeds are the raw
    # nearest-shrunk-mode rule. We reuse the shared helper with an effective ab=1 fallback.
    abar = getattr(proc, "abar", None)
    if abar is not None:
        pred_resp, _ = core.predict_responsibility(X0, means_t, abar, variance,
                                                   delta=cfg.eval.delta)
    else:
        # flow: seeds ~ N(0,I); label by nearest mode direction (responsibility at unit var)
        d2 = torch.cdist(X0, means_t) ** 2
        w = torch.softmax(-d2 / (2.0), 1)
        pmax, arg = w.max(1)
        pred_resp = torch.where(pmax >= 1 - cfg.eval.delta, arg,
                                torch.full_like(arg, -1))
    resp_full = float((pred_resp == gt).float().mean())
    resp_cls = float((pred_resp[m_cls] == gt[m_cls]).float().mean())

    # analytic-field atlas + all-anchor vote, swept over anchor counts
    T_true = cfg.sweep.T_true
    proc_true = make_process(sampler, means_t, variance, T_true, device, cfg)
    m_hall = gt == -1
    n_mode = int(m_cls.sum().item())
    n_hall = int(m_hall.sum().item())
    atlas_rows = []
    for n_disk in cfg.sweep.anchors:
        t0 = time.time()
        anchors, alabels = _build_atlas(proc_true, means_t, R99, K, d, int(n_disk),
                                        cfg.eval.shell_w, device)
        pred_atlas, na, h = core.gauss_vote_all(
            X0, anchors, alabels, K, R99, d=d, h_frac=cfg.eval.h_frac, device=device,
        )
        # full accuracy: exact-label match over ALL points
        full_acc = float((pred_atlas == gt).float().mean())
        # mode accuracy: of all TRUE-mode points, fraction predicted with the CORRECT mode
        #   (wrong mode OR predicted-hallucination both count as wrong)
        mode_acc = float((pred_atlas[m_cls] == gt[m_cls]).float().mean()) if n_mode else float("nan")
        # hallucination accuracy: of all TRUE-hallucination points, fraction predicted -1
        hall_acc = float((pred_atlas[m_hall] == -1).float().mean()) if n_hall else float("nan")
        atlas_rows.append({
            "n_disk": int(n_disk), "n_anchors": int(na), "bandwidth": h,
            "full_acc": full_acc, "class_acc": mode_acc,
            "mode_acc": mode_acc, "hall_acc": hall_acc,
            "secs": round(time.time() - t0, 1),
        })
        del anchors, alabels
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # pick the anchor count with the best FULL accuracy
    best = max(atlas_rows, key=lambda a: a["full_acc"]) if atlas_rows else None

    return {"d": d, "K": K, "hall_gt": hall_gt,
            "n_mode": n_mode, "n_hall": n_hall,
            "responsibility": {"full_acc": resp_full, "class_acc": resp_cls},
            "atlas": atlas_rows,
            "best": best}


def _build_atlas(proc, means_t, R99, K, d, n_disk, shell_w, device):
    """Disk (mode) + shell (hallucination) backtracked via the process' ANALYTIC field."""
    n_shell = max(1, n_disk // 2)
    seeds_all, labs_all = [], []
    for k in range(K):
        u = torch.rand(n_disk, device=device) ** (1.0 / d)
        r_in = R99 * u
        dirs = torch.randn(n_disk, d, device=device)
        dirs /= dirs.norm(dim=1, keepdim=True)
        disk = means_t[k] + r_in[:, None] * dirs
        seeds_all.append(proc.true_field_backtrack(disk))
        labs_all += [k] * n_disk

        r_sh = R99 * (1 + shell_w * torch.rand(n_shell, device=device))
        d2 = torch.randn(n_shell, d, device=device)
        d2 /= d2.norm(dim=1, keepdim=True)
        shell = means_t[k] + r_sh[:, None] * d2
        seeds_all.append(proc.true_field_backtrack(shell))
        labs_all += [-1] * n_shell
    return torch.cat(seeds_all, 0), torch.tensor(labs_all, device=device)


def run(cfg):
    device = core.get_device(cfg.device)
    sampler = cfg.sampler
    out_dir = os.path.join(sampler_dir(cfg.paths.output, sampler))
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            r = eval_one(cfg, int(d), int(K), device)
            if r is None:
                print(f"[eval:{sampler}] d={d:>2} K={K:>2} -> no checkpoint, skipped")
                continue
            results.append(r)
            best = max((a["class_acc"] for a in r["atlas"]), default=float("nan"))
            print(f"[eval:{sampler}] d={d:>2} K={K:>2} hall_gt={r['hall_gt']:.3f} "
                  f"resp_cls={r['responsibility']['class_acc']:.3f} "
                  f"best_atlas_cls={best:.3f}")

    js = os.path.join(out_dir, "results.json")
    with open(js, "w") as f:
        json.dump({"sampler": sampler,
                   "config_sweep": {"d": list(cfg.sweep.d), "K": list(cfg.sweep.K),
                                    "anchors": list(cfg.sweep.anchors),
                                    "T_true": cfg.sweep.T_true},
                   "results": results}, f, indent=2)
    csv = os.path.join(out_dir, "results.csv")
    with open(csv, "w") as f:
        f.write("sampler,d,K,hall_gt,method,n_anchors,bandwidth,full_acc,mode_acc,hall_acc\n")
        for r in results:
            f.write(f"{sampler},{r['d']},{r['K']},{r['hall_gt']:.4f},responsibility,0,0,"
                    f"{r['responsibility']['full_acc']:.4f},"
                    f"{r['responsibility']['class_acc']:.4f},\n")
            for a in r["atlas"]:
                f.write(f"{sampler},{r['d']},{r['K']},{r['hall_gt']:.4f},atlas,{a['n_anchors']},"
                        f"{a['bandwidth']:.4f},{a['full_acc']:.4f},"
                        f"{a['mode_acc']:.4f},{a['hall_acc']:.4f}\n")
    print(f"[eval:{sampler}] wrote {js} and {csv}")
    return {"json": js, "csv": csv, "n_results": len(results)}
