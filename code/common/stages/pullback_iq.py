"""stages/pullback_iq.py — Prop. 4 pulled-back normal vs Intrinsic Quenching (DDIM only).

For each (d, K) in pullback.d x pullback.K, with the trained checkpoint at sweep.T_train:

  1. draw pullback.n_seeds seeds, keep the ones whose endpoint under the field F
     (pullback.field: learned | true) is an intermodal hallucination: farther than
     pullback.deep_mult * R99 from every mode AND projecting strictly between its two
     nearest modes (so a pairwise-channel point, not a tail sample beyond a mode);
     study the first pullback.n_study of them.
  2. OG  : plain DDIM trajectory of each seed.
     IQ  : same seed, score replaced by s - lam grad E, E(x_t) = DSM loss at t0 of the
           Tweedie estimate (Intrinsic Quenching). The energy's denoiser is
           pullback.iq_energy (learned = real IQ, true = exact-GMM LID). lam = 0 picks the
           best-rescuing value from pullback.lam_grid.
     Ours: plain DDIM from z_h - eps* n, with n = J^T nu / ||J^T nu|| (Prop. 4 applied to the
           level set of h_i through z_h, J = DG(z_h)) and eps* the smallest step along -n
           whose endpoint lands inside the certified core C_i. i = the mode IQ reaches
           (or the nearest mode to the OG endpoint if IQ fails).
  3. measures per seed
       eps*, the first-order eps from eq. (11), eps along random directions
       IQ push per step and in total
       z_IQ = G^{-1}(x0_IQ) (step-wise Newton inversion) and Align = cos(z_IQ - z_h, -n)  [eq. 12],
                   measured both at z_h and at the anchor z_a = z_h - eps* n (Prop. 4's Align_a)
       radial vs angular first-order gain at the anchor: <nu, J v> and ||P J v||/r with
                   P = I - nu nu^T, for IQ's direction and for -n (Prop. 4's side-effect term)
       tangent control: steps along v with <J^T nu, v> = 0 should keep the mode (nearest mode
                   unchanged; the anchor sits exactly on the core boundary, so the certified
                   label there is a coin flip) and slide along the core rather than move in or out
       cert time : first t at which the plain sampler from the IQ state lands in C_i
       split time: first t at which the exact-GMM posterior means of IQ (or Ours) and OG
                   are > R99 apart
       deviation curves ||x_t - x_t^OG|| for IQ and Ours

Writes output/<run_tag>/<process>/pullback/d{d}_K{K}.{npz,json} and summary.csv.
Figures: stage pullback_viz -> visualization/<run_tag>/<process>/pullback/
  d{d}_K{K}_stats: deviation curves and seed-space budgets over all studied seeds
  d2_K{K}_example: one seed -- trajectories over the mode classes and the distance from
                   the original trajectory.
  d2_K{K}_anim:    the same seed as an animation (gif, plus mp4 if ffmpeg is present);
                   turn off with pullback.animate=false.

    python code/ddim/main.py sweep=atlas stages=[pullback,pullback_viz]
    python code/ddim/main.py sweep=atlas stages=[pullback,pullback_viz] pullback.d=[2] pullback.K=[2,4]
    python code/baselines/main.py sweep=atlas stages=[pullback] pullback.ckpt_process=ddim
"""
from __future__ import annotations
import os
import csv
import json
import time
import numpy as np
import torch

from common import gmm, checkpoint, utils
from common.process import make_process


def ckpt_process(cfg):
    """Process whose checkpoints and sampler are used: pullback.ckpt_process, else cfg.process."""
    return str(cfg.pullback.ckpt_process or cfg.process)


def out_dir(cfg):
    return os.path.join(cfg.paths.output, cfg.run_tag, cfg.process, "pullback")


def viz_dir(cfg):
    return os.path.join(cfg.paths.viz, cfg.run_tag, cfg.process, "pullback")


# ------------------------------------------------------------------ fields
class Field:
    """eps-prediction at integer level i, differentiable (unlike gmm.true_score)."""

    def __init__(self, proc, model, kind):
        self.p, self.m, self.kind = proc, model, kind

    def eps(self, x, i):
        if self.kind == "learned":
            ti = torch.full((x.shape[0],), int(i), dtype=torch.long, device=x.device)
            return self.m(x, ti)
        ab = self.p.abar[i]
        v = ab * self.p.variance + (1 - ab)
        mu = torch.sqrt(ab) * self.p.means_t
        w = torch.softmax(-torch.cdist(x, mu) ** 2 / (2 * v), 1)
        return -torch.sqrt(1 - ab) * (w @ mu - x) / v


class Sampler:
    """DDIM on the process's schedule. Level T-1 is the seed, level 0 the endpoint."""

    def __init__(self, proc, F):
        self.p, self.F, self.T = proc, F, proc.T
        self.ab = proc.abar

    def step(self, x, i, e=None):
        return self.p._ddim_step(x, i, self.F.eps(x, i) if e is None else e)

    def flow(self, x, i_start):
        for i in range(int(i_start), 0, -1):
            x = self.step(x, i)
        return x

    def G(self, z, chunk=20000):
        with torch.no_grad():
            return torch.cat([self.flow(c, self.T - 1) for c in z.split(chunk)])

    def traj(self, z):
        with torch.no_grad():
            xs = [z]
            for i in range(self.T - 1, 0, -1):
                xs.append(self.step(xs[-1], i))
        return torch.stack(xs)                               # (T, B, d), row k = level T-1-k

    def tweedie(self, x, i):
        return (x - torch.sqrt(1 - self.ab[i]) * self.F.eps(x, i)) / torch.sqrt(self.ab[i])

    def t_of(self, i):
        return (int(i) + 1) / self.T


def level(k, T):
    return T - 1 - k


# ------------------------------------------------------------------ pieces
def pulled_normal(S, x, i_start, mu_tgt):
    """n = J^T nu / ||J^T nu|| for the flow from level i_start; also ||J^T nu|| and G(x)."""
    x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        xo = S.flow(x, i_start)
        nu = (xo - mu_tgt).detach()
        nu = nu / nu.norm(dim=1, keepdim=True)
        (g,) = torch.autograd.grad((xo * nu).sum(), x)
    gn = g.norm(dim=1)
    return g / gn[:, None], gn, xo.detach()


def full_jacobian(S, x, i_start):
    """(B, d, d) Jacobian of the flow from level i_start, by d vector-Jacobian products."""
    d = x.shape[1]
    x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        xo = S.flow(x, i_start)
        rows = [torch.autograd.grad(xo[:, j].sum(), x, retain_graph=j < d - 1)[0]
                for j in range(d)]
    return torch.stack(rows, 1), xo.detach()                 # J[b, out, in]


def make_energy(S, FE, i0, MC):
    ab0 = S.ab[i0]

    def energy(x, i):
        x0 = S.tweedie(x, i)
        B, M, d = x.shape[0], MC.shape[0], x.shape[1]
        xt0 = torch.sqrt(ab0) * x0[:, None, :] + torch.sqrt(1 - ab0) * MC[None]
        r = FE.eps(xt0.reshape(-1, d), i0).reshape(B, M, d) - MC[None]
        return (r ** 2).sum(-1).mean(-1)

    def grad(x, i):
        x = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            (g,) = torch.autograd.grad(energy(x, i).sum(), x)
        return g

    return grad


def sample_iq(S, gradE, z, lam, window):
    xs, push = [z], []
    x = z
    for i in range(S.T - 1, 0, -1):
        with torch.no_grad():
            e0 = S.F.eps(x, i)
            x_plain = S.p._ddim_step(x, i, e0)
        if S.t_of(i) <= window and lam > 0:
            e = e0 + lam * torch.sqrt(1 - S.ab[i]) * gradE(x, i)
            with torch.no_grad():
                xn = S.p._ddim_step(x, i, e)
        else:
            xn = x_plain
        push.append((xn - x_plain).norm(dim=1))
        x = xn.detach()
        xs.append(x)
    return torch.stack(xs), torch.stack(push)                # (T,B,d), (T-1,B)


def first_inside(S, zh, dirs, mu_tgt, R99, grid, iters):
    """Smallest eps with G(zh + eps dir) inside C_tgt: grid scan, then bisection."""
    Bn, Gn, d = zh.shape[0], grid.numel(), zh.shape[1]
    zz = (zh[:, None, :] + grid[None, :, None] * dirs[:, None, :]).reshape(-1, d)
    xo = S.G(zz).reshape(Bn, Gn, d)
    ok = (xo - mu_tgt[:, None, :]).norm(dim=-1) <= R99
    hit = ok.any(1)
    j = ok.float().argmax(1)
    hi = grid[j]
    lo = torch.where(j > 0, grid[(j - 1).clamp(min=0)], torch.zeros_like(hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = (S.G(zh + mid[:, None] * dirs) - mu_tgt).norm(dim=1) <= R99
        hi = torch.where(m, mid, hi)
        lo = torch.where(m, lo, mid)
    return torch.where(hit, hi, torch.full_like(hi, float("nan")))


def invert(S, x0, iters):
    """G^{-1}: undo one DDIM step at a time, each solved by Newton with an autograd Jacobian."""
    x = x0.detach()
    d = x.shape[1]
    for i in range(1, S.T):                                  # recover level i from level i-1
        with torch.no_grad():                                # DDIM-inversion initial guess
            e = S.F.eps(x, i - 1)
            x0h = (x - torch.sqrt(1 - S.ab[i - 1]) * e) / torch.sqrt(S.ab[i - 1])
            y = torch.sqrt(S.ab[i]) * x0h + torch.sqrt(1 - S.ab[i]) * e
        for _ in range(iters):
            y = y.detach().requires_grad_(True)
            with torch.enable_grad():
                out = S.step(y, i)
                J = torch.stack([torch.autograd.grad(out[:, j].sum(), y, retain_graph=j < d - 1)[0]
                                 for j in range(d)], 1)       # (B, d, d), J[b, out_j, in]
            with torch.no_grad():
                y = y - torch.linalg.solve(J, (out - x)[..., None])[..., 0]
        x = y.detach()
    return x


def split_time(S, Ta, Tb, R99):
    """First t (sampling order) at which the posterior means of two trajectories are > R99 apart.

    S should carry the exact-GMM field: its posterior mean is well conditioned at the noise end,
    whereas a learned eps divided by sqrt(abar_T) ~ 7e-3 amplifies small errors ~150x."""
    B = Ta.shape[1]
    out = torch.full((B,), float("nan"), device=Ta.device)
    with torch.no_grad():
        for k in range(S.T):
            i = level(k, S.T)
            far = (S.tweedie(Ta[k], i) - S.tweedie(Tb[k], i)).norm(dim=1) > R99
            out = torch.where(far & torch.isnan(out), torch.full_like(out, S.t_of(i)), out)
    return out


def cert_time(S, Tr, mu_tgt, R99):
    """First t at which the plain sampler, started from the trajectory state, lands in C_tgt."""
    T, B, d = Tr.shape
    with torch.no_grad():
        X = Tr.clone()                                       # row k starts at level T-1-k
        for i in range(T - 1, 0, -1):
            n_act = T - i                                    # rows whose start level >= i
            X[:n_act] = S.step(X[:n_act].reshape(-1, d), i).reshape(n_act, B, d)
        ins = (X - mu_tgt[None]).norm(dim=-1) <= R99         # (T, B)
    t = torch.tensor([S.t_of(level(k, T)) for k in range(T)], device=Tr.device)
    first = ins.float().argmax(0)
    return torch.where(ins.any(0), t[first], torch.full((B,), float("nan"), device=Tr.device))


# ------------------------------------------------------------------ one cell
def pullback_one(cfg, d, K, device):
    pc, pname = cfg.pullback, ckpt_process(cfg)
    path = utils.ckpt_path(cfg.paths.checkpoints, pname, d, K, int(cfg.sweep.T_train))
    if not os.path.exists(path):
        print(f"[pullback] d={d} K={K}: no checkpoint at {path}, skipped")
        return None
    model, ck = checkpoint.load(path, device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    means_t, R99, variance = ck["means"], float(ck["R99"]), ck["variance"]
    proc = make_process(pname, means_t, variance, ck["T"], device, cfg)
    if not hasattr(proc, "_ddim_step"):
        raise RuntimeError("pullback_iq supports the ddim process only")
    S = Sampler(proc, Field(proc, model, str(pc.field)))
    FE = Field(proc, model, str(pc.iq_energy))
    T = S.T
    t0 = time.time()

    # 1. intermodal hallucinations
    Z = proc.seeds(int(pc.n_seeds), d, int(pc.seed) + 17)
    X0 = S.G(Z)
    D = torch.cdist(X0, means_t)
    dmin = D.min(1).values
    hall = dmin > R99
    deep = dmin > float(pc.deep_mult) * R99
    if K >= 2:                                               # between its two nearest modes,
        two = D.topk(2, dim=1, largest=False).indices        # not beyond one (a tail sample)
        m1, m2 = means_t[two[:, 0]], means_t[two[:, 1]]
        ax = m2 - m1
        s_ = ((X0 - m1) * ax).sum(1) / (ax * ax).sum(1)
        deep = deep & (s_ > 0) & (s_ < 1)
    idx = torch.nonzero(deep).flatten()[: int(pc.n_study)]
    Sn = int(idx.numel())
    info = {"d": d, "K": K, "T": T, "R99": R99, "field": str(pc.field),
            "iq_energy": str(pc.iq_energy), "n_seeds": int(pc.n_seeds),
            "hall_rate": float(hall.float().mean()), "deep_rate": float(deep.float().mean()),
            "n_study": Sn}
    print(f"[pullback] d={d:>2} K={K:>2} HR={100*info['hall_rate']:.2f}% "
          f"intermodal={100*info['deep_rate']:.2f}% -> studying {Sn}", flush=True)
    if Sn == 0:
        return info
    zh = Z[idx]

    # 2. OG and IQ
    T_og = S.traj(zh)
    i0 = int(np.clip(round(float(pc.iq_t0) * T - 1), 0, T - 1))
    g = torch.Generator(device=device).manual_seed(int(pc.seed) + 5)
    MC = torch.randn(int(pc.n_mc) // 2, d, generator=g, device=device)
    MC = torch.cat([MC, -MC])                                # antithetic: no side bias
    gradE = make_energy(S, FE, i0, MC)
    lab = lambda x: gmm.label_fate(x, means_t, R99)
    lam, scan = float(pc.lam), {}
    if lam <= 0:
        for l_ in pc.lam_grid:
            Tq, _ = sample_iq(S, gradE, zh, float(l_), float(pc.iq_window))
            scan[float(l_)] = float((lab(Tq[-1]) >= 0).float().mean())
        lam = max(scan, key=lambda l_: (scan[l_], -l_))
        print(f"[pullback]   IQ lambda scan {scan} -> {lam}", flush=True)
    T_iq, PUSH = sample_iq(S, gradE, zh, lam, float(pc.iq_window))
    lab_iq = lab(T_iq[-1])
    x0h = T_og[-1]
    near = torch.cdist(x0h, means_t).argmin(1)
    tgt = torch.where(lab_iq >= 0, lab_iq, near)
    mu_t = means_t[tgt]

    # 3. pulled-back normal and Ours
    nrm, gn, _ = pulled_normal(S, zh, T - 1, mu_t)
    rad = (x0h - mu_t).norm(dim=1)
    eps_lin = (rad ** 2 - R99 ** 2) / (2 * rad * gn)          # eq. (11) at z_h
    grid = torch.logspace(-4, float(np.log10(pc.eps_max)), int(pc.n_eps_grid), device=device)
    eps_star = first_inside(S, zh, -nrm, mu_t, R99, grid, 30)
    found = ~torch.isnan(eps_star)
    z_ours = zh - torch.nan_to_num(eps_star)[:, None] * nrm
    T_ours = S.traj(z_ours)
    lab_ours = torch.where(found, lab(T_ours[-1]), torch.full_like(lab_iq, -2))

    eps_rand = torch.full((Sn, int(pc.n_rand)), float("nan"), device=device)
    for r in range(int(pc.n_rand)):
        u = torch.randn(Sn, d, generator=g, device=device)
        u = u / u.norm(dim=1, keepdim=True)
        eps_rand[:, r] = first_inside(S, zh, u, mu_t, R99, grid, 12)

    # 3b. anchor geometry: radial vs angular gains, tangent control (Prop. 4)
    anchor = dict(align=np.nan, ang_rad_iq=np.nan, ang_rad_ours=np.nan,
                  tang_same=np.nan, tang_radial_ratio=np.nan, tang_arc=np.nan)
    z_a = zh[found] - eps_star[found][:, None] * nrm[found]
    if int(found.sum()):
        Ja, xa = full_jacobian(S, z_a, T - 1)
        mu_a = means_t[tgt[found]]
        r_a = (xa - mu_a).norm(dim=1, keepdim=True)
        nu_a = (xa - mu_a) / r_a
        g_a = torch.einsum("boi,bo->bi", Ja, nu_a)           # J^T nu
        gn_a = g_a.norm(dim=1)
        n_a = g_a / gn_a[:, None]

        def gains(v):                                        # v: (B, d) unit seed direction
            Jv = torch.einsum("boi,bi->bo", Ja, v)
            radial = (nu_a * Jv).sum(1)                      # d(terminal radius)
            tang = Jv - radial[:, None] * nu_a               # P Jv
            return radial, tang.norm(dim=1) / r_a[:, 0]      # radial, angular (rad per unit step)

        rad_n, ang_n = gains(-n_a)
        anchor["ang_rad_ours"] = float((ang_n / rad_n.abs()).median())
        # tangent control: step along v with <J^T nu, v> = 0, by the same eps*
        n_tan = int(pc.n_tan)
        same, dr_ratio, arc = [], [], []
        for _ in range(n_tan):
            u = torch.randn(z_a.shape[0], d, generator=g, device=device)
            v = u - (u * n_a).sum(1, keepdim=True) * n_a
            v = v / v.norm(dim=1, keepdim=True)
            dz = float(pc.tang_mult) * eps_star[found][:, None]
            xt_ = S.G(z_a + dz * v)
            r_t = (xt_ - mu_a).norm(dim=1)
            same.append((torch.cdist(xt_, means_t).argmin(1) == tgt[found]).float().mean())
            dr_ratio.append(((r_t - r_a[:, 0]).abs() /
                             (rad_n.abs() * dz[:, 0]).clamp_min(1e-12)).median())
            cos = ((xt_ - mu_a) * nu_a).sum(1) / r_t.clamp_min(1e-12)
            arc.append((r_a[:, 0] * torch.arccos(cos.clamp(-1, 1))).median())
        anchor["tang_same"] = float(torch.stack(same).mean())
        anchor["tang_radial_ratio"] = float(torch.stack(dr_ratio).median())
        anchor["tang_arc"] = float(torch.stack(arc).median())

    # 4. IQ in seed space, per-step alignment, timing
    z_iq = invert(S, T_iq[-1], int(pc.newton_iters))
    inv_err = (S.G(z_iq) - T_iq[-1]).norm(dim=1)
    v_iq = z_iq - zh
    align = (v_iq * -nrm).sum(1) / v_iq.norm(dim=1)
    if int(found.sum()):
        v_u = v_iq[found] / v_iq[found].norm(dim=1, keepdim=True)
        anchor["align"] = float((v_u * -n_a).sum(1).median())
        rad_iq, ang_iq = gains(v_u)
        anchor["ang_rad_iq"] = float((ang_iq / rad_iq.abs()).median())

    cert = cert_time(S, T_iq, mu_t, R99)
    S_ref = Sampler(proc, Field(proc, model, "true"))
    split_iq = split_time(S_ref, T_iq, T_og, R99)
    split_ours = split_time(S_ref, T_ours, T_og, R99)
    tgrid = np.array([S.t_of(level(k, T)) for k in range(T)])

    def med(x):
        x = x[~torch.isnan(x)]
        return float(x.median()) if x.numel() else None

    resc = lab_iq >= 0
    info.update({
        "lam": lam, "lam_scan": scan, "iq_window": float(pc.iq_window), "iq_t0_level": i0,
        "iq_rescue": float(resc.float().mean()),
        "ours_found": float(found.float().mean()),
        "ours_rescue": float((lab_ours >= 0).float().mean()),
        "ours_same_mode_as_iq": float(((lab_ours == lab_iq) & resc).sum()) / max(1, int(resc.sum())),
        "eps_ours_med": med(eps_star), "eps_lin_med": med(eps_lin),
        "eps_ratio_lin_med": med(eps_star / eps_lin),
        "eps_rand_med": med(eps_rand.flatten()),
        "rand_fail": float(torch.isnan(eps_rand).float().mean()),
        "iq_push_per_step_mean": float(PUSH.mean()),
        "iq_push_total_med": med(PUSH.sum(0)),
        "iq_seed_shift_med": med(v_iq.norm(dim=1)),
        "align_all_med": med(align), "align_resc_med": med(align[resc]),
        "align_anchor_med": anchor["align"],
        "ang_over_rad_iq": anchor["ang_rad_iq"], "ang_over_rad_ours": anchor["ang_rad_ours"],
        "tang_same_mode": anchor["tang_same"], "tang_radial_ratio": anchor["tang_radial_ratio"],
        "tang_arc_med": anchor["tang_arc"],
        "inv_err_max": float(inv_err.max()),
        "cert_t_iq_med": med(cert), "split_t_iq_med": med(split_iq),
        "split_t_ours_med": med(torch.where(found, split_ours, torch.full_like(split_ours, float("nan")))),
        "secs": round(time.time() - t0, 1),
    })

    os.makedirs(out_dir(cfg), exist_ok=True)
    cpu = lambda t: t.detach().cpu().numpy()
    Dog = lambda Tr: cpu((Tr - T_og).norm(dim=-1))
    np.savez_compressed(
        os.path.join(out_dir(cfg), f"d{d}_K{K}.npz"),
        t=tgrid, means=cpu(means_t), R99=R99, zh=cpu(zh), z_ours=cpu(z_ours), z_iq=cpu(z_iq),
        nrm=cpu(nrm), T_og=cpu(T_og), T_iq=cpu(T_iq), T_ours=cpu(T_ours), push=cpu(PUSH),
        D_iq=Dog(T_iq), D_ours=Dog(T_ours), D_iq_ours=cpu((T_iq - T_ours).norm(dim=-1)),
        split_iq=cpu(split_iq), split_ours=cpu(split_ours), cert_iq=cpu(cert),
        align=cpu(align), eps_star=cpu(eps_star),
        eps_lin=cpu(eps_lin), eps_rand=cpu(eps_rand), lab_iq=cpu(lab_iq),
        lab_ours=cpu(lab_ours), tgt=cpu(tgt), inv_err=cpu(inv_err))
    with open(os.path.join(out_dir(cfg), f"d{d}_K{K}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[pullback]   IQ rescue {100*info['iq_rescue']:.0f}%  Ours {100*info['ours_rescue']:.0f}%  "
          f"same mode {100*info['ours_same_mode_as_iq']:.0f}%  Align {info['align_resc_med']}  "
          f"Align_a {info['align_anchor_med']}  ang/rad IQ {info['ang_over_rad_iq']}  "
          f"tangent keeps mode {info['tang_same_mode']}  "
          f"eps* {info['eps_ours_med']}  IQ shift {info['iq_seed_shift_med']}  "
          f"cert {info['cert_t_iq_med']}  split IQ/Ours {info['split_t_iq_med']}/{info['split_t_ours_med']}  "
          f"({info['secs']}s)", flush=True)
    return info


SUMMARY_KEYS = ["d", "K", "n_study", "hall_rate", "deep_rate", "lam", "iq_rescue", "ours_rescue",
                "ours_same_mode_as_iq", "eps_ours_med", "eps_lin_med", "eps_rand_med", "rand_fail",
                "iq_seed_shift_med", "iq_push_per_step_mean", "iq_push_total_med", "align_resc_med",
                "align_anchor_med", "ang_over_rad_iq", "ang_over_rad_ours", "tang_same_mode",
                "tang_radial_ratio", "tang_arc_med", "cert_t_iq_med", "split_t_iq_med", "split_t_ours_med", "inv_err_max"]


def run(cfg):
    device = utils.get_device(cfg.device)
    rows = []
    for d in cfg.pullback.d:
        for K in cfg.pullback.K:
            r = pullback_one(cfg, int(d), int(K), device)
            if r is not None and r.get("n_study", 0) > 0:
                rows.append(r)
    os.makedirs(out_dir(cfg), exist_ok=True)
    path = os.path.join(out_dir(cfg), "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_KEYS)
        for r in rows:
            w.writerow([r.get(k) for k in SUMMARY_KEYS])
    print(f"[pullback] wrote {path}")
    return {"summary": path, "n_cells": len(rows)}


# ------------------------------------------------------------------ figures (stage pullback_viz)
C_OG, C_IQ, C_US = "#7f7f7f", "#1f77b4", "#d62728"


def _stats_fig(D, J, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = D["t"]
    ok = D["lab_ours"] >= 0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for key, col, name in (("D_iq", C_IQ, "IQ"), ("D_ours", C_US, "Ours")):
        M = D[key][:, ok] if key == "D_ours" else D[key]
        for j in range(M.shape[1]):
            ax[0].plot(t, M[:, j], color=col, alpha=0.08, lw=0.8)
        ax[0].plot(t, np.median(M, 1), color=col, lw=2.2, label=f"{name} vs OG")
    si, so = np.nanmedian(D["split_iq"]), np.nanmedian(D["split_ours"][ok])
    ax[0].axvline(si, color=C_IQ, ls="--", lw=1)
    ax[0].axvline(so, color=C_US, ls="--", lw=1)
    ax[0].plot(t, np.median(D["D_iq_ours"][:, ok], 1), color="k", lw=1.4, ls=":", label="IQ vs Ours")
    ax[0].set_yscale("log"); ax[0].invert_xaxis()
    ax[0].set_xlabel("t  (sampling runs 1 → 0)"); ax[0].set_ylabel(r"$\|x_t - x_t^{OG}\|$")
    ax[0].set_title(f"Deviation from OG (dashed: split, IQ {si:.3f}, Ours {so:.3f})", fontsize=10)
    ax[0].legend(fontsize=9, loc="upper left")

    vals = [D["eps_star"], D["eps_lin"], D["eps_rand"].ravel(), np.linalg.norm(D["z_iq"] - D["zh"], axis=1)]
    names = ["Ours\n(along $-n$)", "Ours\n1st order", "random dirs\n(successes)", "IQ\nseed shift"]
    bp = ax[1].boxplot([v[np.isfinite(v)] for v in vals], showfliers=False, patch_artist=True)
    ax[1].set_xticks(range(1, 5)); ax[1].set_xticklabels(names)
    for b_, c_ in zip(bp["boxes"], [C_US, "#f4a582", "0.7", C_IQ]):
        b_.set_facecolor(c_); b_.set_alpha(0.6)
    ax[1].set_yscale("log"); ax[1].set_ylabel(r"seed displacement $\|\Delta z\|$")
    ar = J["align_resc_med"]
    ax[1].set_title(f"Seed-space budget\nAlign(v_IQ), IQ-rescued: {ar if ar is None else round(ar, 3)};  "
                    f"random dirs failing: {100*J['rand_fail']:.0f}%", fontsize=10)
    fig.suptitle(f"d={J['d']}, K={J['K']}, field={J['field']}, IQ energy={J['iq_energy']}, "
                 f"lambda={J['lam']}:  {J['n_study']} intermodal seeds.  IQ rescues "
                 f"{100*J['iq_rescue']:.0f}%, Ours {100*J['ours_rescue']:.0f}%, same mode "
                 f"{100*J['ours_same_mode_as_iq']:.0f}%", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=170)
    plt.close(fig)


def _manual_fig(D, J, path):
    """d = 2: trajectories over the data-space classes, and the distance from the original."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    K, R99, MU, t = J["K"], float(D["R99"]), D["means"], D["t"]
    ok = (D["lab_iq"] >= 0) & (D["lab_ours"] >= 0)
    j = int(np.flatnonzero(ok)[0]) if ok.any() else int(np.flatnonzero(D["lab_ours"] >= 0)[0])
    pale = ["#fde0dd", "#deebf7", "#e5f5e0", "#fff7bc", "#efedf5", "#fee6ce", "#e0f3f8",
            "#f2f0f7", "#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc",
            "#e5d8bd", "#fddaec"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- left: data space, classes + cores + the three trajectories ------------------
    Tr = {k: D[k][:, j] for k in ("T_og", "T_iq", "T_ours")}
    pts = np.concatenate(list(Tr.values()) + [MU])
    lo, hi = pts.min(0), pts.max(0)
    c, w = 0.5 * (lo + hi), 0.62 * max(hi - lo) + 2 * R99
    gx = np.linspace(c[0] - w, c[0] + w, 400)
    gy = np.linspace(c[1] - w, c[1] + w, 400)
    GX, GY = np.meshgrid(gx, gy)
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    near = np.argmin(((P[:, None, :] - MU[None]) ** 2).sum(-1), 1).reshape(GX.shape)
    ax[0].pcolormesh(GX, GY, near, cmap=ListedColormap(pale[:K]), shading="auto",
                     vmin=0, vmax=K - 1, alpha=0.65)
    ax[0].contour(GX, GY, near, levels=np.arange(K) + 0.5, colors="0.55",
                  linestyles="--", linewidths=0.9)
    th = np.linspace(0, 2 * np.pi, 240)
    for k in range(K):
        ax[0].plot(MU[k, 0] + R99 * np.cos(th), MU[k, 1] + R99 * np.sin(th), color="0.35", lw=1.2)
        ax[0].plot(*MU[k], "k*", ms=12)
        ax[0].annotate(rf"$\mu_{{{k}}}$", MU[k], textcoords="offset points", xytext=(8, 6),
                       fontsize=11)
    for key, col, name in (("T_og", C_OG, "original (hallucinates)"), ("T_iq", C_IQ, "IQ"),
                           ("T_ours", C_US, "ours (seed moved along $-n$)")):
        ax[0].plot(Tr[key][:, 0], Tr[key][:, 1], color=col, lw=1.8, label=name, zorder=3)
        ax[0].plot(*Tr[key][-1], "o", color=col, ms=9, mec="k", zorder=4)
    ax[0].plot(*Tr["T_og"][0], "X", color="k", ms=11, zorder=4)
    ax[0].annotate(r"$x_T$", Tr["T_og"][0], textcoords="offset points", xytext=(9, -14), fontsize=11)
    ax[0].set_xlim(c[0] - w, c[0] + w); ax[0].set_ylim(c[1] - w, c[1] + w)
    ax[0].set_aspect("equal"); ax[0].set_xlabel("$x_1$"); ax[0].set_ylabel("$x_2$")
    ax[0].set_title("Trajectories over the mode classes\n"
                    r"(dashed: class boundaries, circles: cores $r^\bullet=R_{99}$)", fontsize=10)
    ax[0].legend(fontsize=9, loc="best", framealpha=0.92, facecolor="white")

    # --- right: distance from the original trajectory ---------------------------------
    for key, col, name in (("D_iq", C_IQ, "IQ"), ("D_ours", C_US, "ours")):
        M = D[key]
        keep = np.flatnonzero(D["lab_ours"] >= 0) if key == "D_ours" else np.arange(M.shape[1])
        for i2 in keep:
            ax[1].plot(t, M[:, i2], color=col, alpha=0.12, lw=0.8)
        ax[1].plot(t, M[:, j], color=col, lw=2.2, label=f"{name} vs original")
    for key, col in (("split_iq", C_IQ), ("split_ours", C_US)):
        if np.isfinite(D[key][j]):
            ax[1].axvline(D[key][j], color=col, ls="--", lw=1)
    ax[1].axhline(R99, color="0.5", lw=1, ls=":", label=r"$R_{99}$")
    ax[1].set_yscale("log"); ax[1].invert_xaxis()
    ax[1].set_xlabel("t  (sampling runs 1 → 0)")
    ax[1].set_ylabel(r"$\|x_t-x_t^{\mathrm{orig}}\|$")
    ax[1].set_title("Distance from the original trajectory\n"
                    "(bold: the seed on the left, thin: the other studied seeds)", fontsize=10)
    ax[1].legend(fontsize=9, loc="upper left", framealpha=0.92, facecolor="white")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}_example.{ext}", dpi=170)
    plt.close(fig)


def _anim(D, J, path, fps=25, hold=20, max_frames=120, dpi=110):
    """d = 2: the three trajectories drawing themselves, with the distance panel filling in."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    from matplotlib.colors import ListedColormap
    import shutil
    K, R99, MU, t = J["K"], float(D["R99"]), D["means"], D["t"]
    ok = (D["lab_iq"] >= 0) & (D["lab_ours"] >= 0)
    if not ok.any():
        return None
    j = int(np.flatnonzero(ok)[0])
    Tr = {k: D[k][:, j] for k in ("T_og", "T_iq", "T_ours")}
    n = len(t)
    pale = ["#fde0dd", "#deebf7", "#e5f5e0", "#fff7bc", "#efedf5", "#fee6ce", "#e0f3f8",
            "#f2f0f7", "#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc",
            "#e5d8bd", "#fddaec"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))

    pts = np.concatenate(list(Tr.values()) + [MU])
    lo, hi = pts.min(0), pts.max(0)
    c, w = 0.5 * (lo + hi), 0.62 * max(hi - lo) + 2 * R99
    gx, gy = np.linspace(c[0] - w, c[0] + w, 300), np.linspace(c[1] - w, c[1] + w, 300)
    GX, GY = np.meshgrid(gx, gy)
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    near = np.argmin(((P[:, None, :] - MU[None]) ** 2).sum(-1), 1).reshape(GX.shape)
    ax[0].pcolormesh(GX, GY, near, cmap=ListedColormap(pale[:K]), shading="auto",
                     vmin=0, vmax=K - 1, alpha=0.65)
    ax[0].contour(GX, GY, near, levels=np.arange(K) + 0.5, colors="0.55",
                  linestyles="--", linewidths=0.9)
    th = np.linspace(0, 2 * np.pi, 240)
    for k in range(K):
        ax[0].plot(MU[k, 0] + R99 * np.cos(th), MU[k, 1] + R99 * np.sin(th), color="0.35", lw=1.2)
        ax[0].plot(*MU[k], "k*", ms=12)
    ax[0].plot(*Tr["T_og"][0], "X", color="k", ms=11, zorder=5)
    ax[0].set_xlim(c[0] - w, c[0] + w); ax[0].set_ylim(c[1] - w, c[1] + w)
    ax[0].set_aspect("equal"); ax[0].set_xlabel("$x_1$"); ax[0].set_ylabel("$x_2$")

    names = {"T_og": "original (hallucinates)", "T_iq": "IQ", "T_ours": "ours"}
    cols = {"T_og": C_OG, "T_iq": C_IQ, "T_ours": C_US}
    lines = {k: ax[0].plot([], [], color=cols[k], lw=2, label=names[k], zorder=3)[0] for k in Tr}
    dots = {k: ax[0].plot([], [], "o", color=cols[k], ms=9, mec="k", zorder=4)[0] for k in Tr}
    ax[0].legend(fontsize=9, loc="best", framealpha=0.92, facecolor="white")

    for key, col, name in (("D_iq", C_IQ, "IQ"), ("D_ours", C_US, "ours")):
        ax[1].plot(t, D[key][:, j], color=col, lw=1, alpha=0.18)
    dl = {k: ax[1].plot([], [], color=c_, lw=2.2, label=f"{nm} vs original")[0]
          for k, c_, nm in (("D_iq", C_IQ, "IQ"), ("D_ours", C_US, "ours"))}
    ax[1].axhline(R99, color="0.5", lw=1, ls=":", label=r"$R_{99}$")
    now = ax[1].axvline(t[0], color="0.3", lw=1)
    dmax = max(D["D_iq"][:, j].max(), D["D_ours"][:, j].max())
    dmin = max(1e-4, min(D["D_iq"][1:, j].min(), D["D_ours"][:, j].min()))
    ax[1].set_yscale("log"); ax[1].set_ylim(0.5 * dmin, 2 * dmax)
    ax[1].set_xlim(t[0], t[-1])
    ax[1].set_xlabel("t  (sampling runs 1 → 0)")
    ax[1].set_ylabel(r"$\|x_t-x_t^{\mathrm{orig}}\|$")
    ax[1].legend(fontsize=9, loc="upper left", framealpha=0.92, facecolor="white")
    title = fig.suptitle("")

    idx = (np.unique(np.linspace(0, n - 1, max_frames).astype(int)) if n > max_frames
           else np.arange(n))

    def frame(f):
        i = int(idx[min(f, len(idx) - 1)])
        for k in Tr:
            lines[k].set_data(Tr[k][: i + 1, 0], Tr[k][: i + 1, 1])
            dots[k].set_data([Tr[k][i, 0]], [Tr[k][i, 1]])
        for k in dl:
            dl[k].set_data(t[: i + 1], D[k][: i + 1, j])
        now.set_xdata([t[i], t[i]])
        title.set_text(f"d={J['d']}, K={J['K']}:  reverse step {i}/{n-1}   (t = {t[i]:.3f})")
        return list(lines.values()) + list(dots.values()) + list(dl.values()) + [now, title]

    fig.tight_layout()
    an = FuncAnimation(fig, frame, frames=len(idx) + hold, interval=1000 / fps, blit=False)
    if shutil.which("ffmpeg"):
        an.save(f"{path}_anim.mp4", writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
    an.save(f"{path}_anim.gif", writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path + "_anim"


def viz(cfg):
    import glob
    os.makedirs(viz_dir(cfg), exist_ok=True)
    made = []
    for f in sorted(glob.glob(os.path.join(out_dir(cfg), "d*_K*.npz"))):
        D = dict(np.load(f))
        with open(f[:-4] + ".json") as fh:
            J = json.load(fh)
        stem = os.path.join(viz_dir(cfg), os.path.basename(f)[:-4])
        _stats_fig(D, J, stem + "_stats")
        made.append(stem + "_stats")
        if J["d"] == 2:
            _manual_fig(D, J, stem)
            made.append(stem + "_example")
            if bool(cfg.pullback.animate):
                a_ = _anim(D, J, stem, max_frames=int(cfg.pullback.anim_frames))
                if a_:
                    made.append(a_)
    print(f"[pullback_viz] wrote {len(made)} figures under {viz_dir(cfg)}")
    return {"figures": made}