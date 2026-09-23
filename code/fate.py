"""fate.py — predictors of a seed's fate, fit on backtracked anchors, and their scores.

Classes: -1 = hallucination, 0..K-1 = the modes. Every predictor maps seeds (n, d) to
K+1 logits, column 0 the hallucination class.

    knn          vote of the k nearest anchors (seed space)
    altered_knn  weighted kNN over MODE-only anchors (core.altered_knn_anchors); a seed whose
                 vote is low-confidence (normalised entropy) is a hallucination
    quadratic    linear classifier on (x, all pairwise products x_i x_j)
    polar        degree-p polynomial in (direction, radius) giving K mode logits; the
                 hallucination logit is c + a (r - sqrt d) - beta (top1 - top2 mode logit),
                 i.e. "a hallucination lies within a radius-dependent margin of a mode boundary"
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn

PARAMETRIC = ("quadratic", "polar")
NONPARAMETRIC = ("knn", "altered_knn")


def _chunk(n_stored):
    return max(16, int(2e8 // max(n_stored, 1)))       # ~800 MB of distances per block


class KNN(nn.Module):
    """k-nearest-anchor vote; forward returns log class frequencies."""

    def __init__(self, K, k=10):
        super().__init__()
        self.arch, self.K, self.k = "knn", K, int(k)
        self.register_buffer("X", torch.zeros(0))
        self.register_buffer("Y", torch.zeros(0))

    @torch.no_grad()
    def fit(self, X, y):
        self.X, self.Y = X, nn.functional.one_hot(y + 1, self.K + 1).float()
        return self

    @torch.no_grad()
    def forward(self, x):
        out = []
        for s in range(0, x.shape[0], _chunk(self.X.shape[0])):
            D = torch.cdist(x[s:s + _chunk(self.X.shape[0])], self.X)
            idx = D.topk(min(self.k, D.shape[1]), dim=1, largest=False).indices
            out.append(torch.log(self.Y[idx].mean(1) + 1e-9))
        return torch.cat(out)


class AlteredKNN(nn.Module):
    """Weighted kNN over mode-labelled anchors; hallucination = low confidence.

    The k nearest anchors vote s_k = sum_i w_i [y_i = k] / sum_i w_i, p = softmax(s / temperature)
    (p = s for temperature <= 0). Confidence is 1 - H(p) / log K ('entropy') or max p ('max');
    a seed is a hallucination iff confidence < threshold ('auto': fit by calibrate())."""

    def __init__(self, K, k=10, temperature=0.1, threshold=0.5, confidence="entropy"):
        super().__init__()
        self.arch, self.K, self.k = "altered_knn", K, int(k)
        self.temperature, self.confidence = float(temperature), confidence
        self.threshold = None if threshold == "auto" else float(threshold)
        self.register_buffer("X", torch.zeros(0))
        self.register_buffer("Y", torch.zeros(0))
        self.register_buffer("W", torch.zeros(0))

    @torch.no_grad()
    def fit(self, X, y, w=None):
        keep = y >= 0                                  # mode anchors only
        X, y = X[keep], y[keep]
        self.W = torch.ones(X.shape[0], device=X.device) if w is None else w[keep].float()
        self.X, self.Y = X, nn.functional.one_hot(y, self.K).float()
        return self

    @torch.no_grad()
    def conf(self, x):
        """(class probabilities, confidence) of seeds x."""
        out = []
        for s in range(0, x.shape[0], _chunk(self.X.shape[0])):
            D = torch.cdist(x[s:s + _chunk(self.X.shape[0])], self.X)
            idx = D.topk(min(self.k, D.shape[1]), dim=1, largest=False).indices
            wv = self.W[idx]
            votes = (wv[:, :, None] * self.Y[idx]).sum(1) / wv.sum(1, keepdim=True).clamp_min(1e-12)
            out.append(torch.softmax(votes / self.temperature, 1) if self.temperature > 0 else votes)
        p = torch.cat(out)
        if self.confidence == "max":
            return p, p.max(1).values
        H = -(p * torch.log(p.clamp_min(1e-12))).sum(1)
        return p, 1 - H / math.log(max(self.K, 2))

    @torch.no_grad()
    def calibrate(self, X_cal, y_cal):
        """threshold := the confidence cut maximising hallucination F1 on labelled seeds."""
        _, c = self.conf(X_cal)
        hall = y_cal == -1
        best, best_f1 = 0.0, -1.0
        for th in torch.quantile(c, torch.linspace(0.005, 0.5, 100, device=c.device)).tolist():
            pred = c < th
            tp, fp, fn = int((pred & hall).sum()), int((pred & ~hall).sum()), int((~pred & hall).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best, best_f1 = th, f1
        self.threshold = best
        return self

    @torch.no_grad()
    def forward(self, x):
        if self.threshold is None:
            raise RuntimeError("altered_knn with threshold=auto needs calibrate() first")
        p, conf = self.conf(x)
        zh = torch.where(conf < self.threshold, torch.full_like(conf, 30.0), torch.full_like(conf, -30.0))
        return torch.cat([zh[:, None], torch.log(p.clamp_min(1e-12))], 1)


class FateNet(nn.Module):
    """The parametric predictors (quadratic, polar); see the module docstring."""

    def __init__(self, d, K, arch="polar", hidden=256, degree=3):
        super().__init__()
        if arch not in PARAMETRIC:
            raise ValueError(f"unknown parametric arch {arch!r}; choose from {PARAMETRIC}")
        self.arch, self.d, self.K, self.degree = arch, d, K, degree
        if arch == "quadratic":
            self.register_buffer("iu", torch.triu_indices(d, d))
            self.net = nn.Linear(d + self.iu.shape[1], K + 1)
        else:
            self.proj = nn.Linear(d + 1, hidden)
            self.net = nn.Linear(hidden * degree, K)
            self.c = nn.Parameter(torch.zeros(()))
            self.a = nn.Parameter(torch.zeros(()))
            self.log_beta = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        if self.arch == "quadratic":
            return self.net(torch.cat([x, x[:, self.iu[0]] * x[:, self.iu[1]]], 1))
        mu = math.sqrt(self.d)                                     # typical seed radius
        r = x.norm(dim=1, keepdim=True)
        rs = r - mu
        z = self.proj(torch.cat([x / r * math.sqrt(mu), rs], 1)) / math.sqrt(mu + 1)
        zm = self.net(torch.cat([z ** q for q in range(1, self.degree + 1)], 1))   # mode logits
        top = zm.topk(2, dim=1).values
        zh = self.c + self.a * rs[:, 0] - self.log_beta.exp() * (top[:, 0] - top[:, 1])
        return torch.cat([zh[:, None], zm], 1)


def train_fate_classifier(X, y, K, arch, hidden=256, degree=3, epochs=30, batch=2048, lr=3e-3,
                          weight_decay=1e-4, min_steps=0, k=10, temperature=0.1, threshold=0.5,
                          confidence="entropy", w=None, seed=0, device=None):
    """Fit one predictor on seeds X (n, d) with fates y. Non-parametric ones store the anchors;
    parametric ones run AdamW + one-cycle lr for `epochs` passes, at least `min_steps` steps."""
    if arch == "altered_knn":
        return AlteredKNN(K, k, temperature, threshold, confidence).to(device).fit(X, y, w)
    if arch == "knn":
        return KNN(K, k).to(device).fit(X, y)
    torch.manual_seed(seed)
    net = FateNet(X.shape[1], K, arch, hidden, degree).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = X.shape[0]
    batch = min(batch, n)
    per_epoch = max(1, n // batch)
    epochs = max(epochs, -(-min_steps // per_epoch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * per_epoch)
    yl = y + 1                                                    # class index: 0 = hallucination
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n - batch + 1, batch):
            i = perm[s:s + batch]
            loss = nn.functional.cross_entropy(net(X[i]), yl[i])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net


def train_ensemble(X, y, K, spec: dict, device=None, w=None):
    """spec['ensemble'] predictors from one conf/classifier spec (one if non-parametric);
    w = anchor weights (altered_knn only)."""
    kw = dict(arch=spec["arch"], hidden=spec["hidden"], degree=spec["degree"],
              epochs=spec["epochs"], batch=spec["batch"], lr=spec["lr"],
              weight_decay=spec["weight_decay"], min_steps=spec["min_steps"], k=spec["k"],
              temperature=spec["temperature"], threshold=spec["threshold"],
              confidence=spec["confidence"], w=w if spec["arch"] == "altered_knn" else None,
              device=device)
    n_ens = 1 if spec["arch"] in NONPARAMETRIC else spec["ensemble"]
    return [train_fate_classifier(X, y, K, seed=e, **kw) for e in range(n_ens)]


@torch.no_grad()
def predict_fate(nets, X, chunk=100000):
    """Ensemble prediction (summed log-probabilities) -> fates in {-1, 0..K-1}."""
    quad = [int(n.iu.shape[1]) for n in nets if n.arch == "quadratic"]
    if quad:                     # d(d+1)/2 pairwise features: keep a block's features under ~2 GB
        chunk = max(1024, min(chunk, int(2e9 // (4 * max(quad)))))
    out = []
    for s in range(0, X.shape[0], chunk):
        lp = sum(torch.log_softmax(net(X[s:s + chunk]), 1) for net in nets)
        best, k = lp[:, 1:].max(1)
        out.append(torch.where(lp[:, 0] - best > 0, torch.full_like(k, -1), k))
    return torch.cat(out)


# --------------------------------------------------------------------------------------
# scores of predicted vs ground-truth fates
# --------------------------------------------------------------------------------------
METRICS = ("full_acc", "mode_acc", "mode_f1", "hall_prec", "hall_rec", "hall_f1", "balanced")


def _f1(pred_pos, gt_pos):
    tp = int((pred_pos & gt_pos).sum()); fp = int((pred_pos & ~gt_pos).sum())
    fn = int((~pred_pos & gt_pos).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def fate_metrics(pred, gt) -> dict:
    """full_acc: all seeds; mode_acc / mode_f1 (macro): seeds whose true fate is a mode;
    hall_*: the hallucination class; balanced: mean of mode_acc and hall_f1."""
    m_mode = gt >= 0
    mode = float((pred[m_mode] == gt[m_mode]).float().mean()) if m_mode.any() else float("nan")
    prec, rec, f1 = _f1(pred == -1, gt == -1)
    modes = torch.unique(gt[m_mode]).tolist()
    mode_f1 = float(sum(_f1(pred == k, gt == k)[2] for k in modes) / max(len(modes), 1))
    return {"full_acc": float((pred == gt).float().mean()), "mode_acc": mode, "mode_f1": mode_f1,
            "hall_prec": prec, "hall_rec": rec, "hall_f1": f1,
            "balanced": 0.5 * (mode + f1) if mode == mode else float("nan")}
