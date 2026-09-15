"""common/fate.py — seed -> fate classifiers.

The fate of a seed under a deterministic sampler (DDIM, ODE flow) is a function of the
seed, so it can be learned directly: label n seeds by running the sampler once, fit a
classifier seed -> {hallucination, mode_0 .. mode_{K-1}}, predict unseen seeds.

`arch` fixes the geometry of the decision boundaries in seed space (a capacity ladder):

  linear    : W x + b over K+1 classes             -> every boundary is a hyperplane
  margin    : mode logits W x + b (K hyperplanes); hallucination logit
              c - beta * (top1 - top2 mode logit)  -> "hallucinate iff within a margin of a
              linear mode boundary"
  radial    : linear over [x, |x|^2]               -> affine cones + a norm term
  quadratic : linear over [x, x_i x_j (i<=j)]      -> boundaries are quadrics
  poly      : degree-p polynomial in x (Waring form: h = W x + b, readout linear in
              [h, h^2, .., h^p]; spans all degree<=p polynomials for large `hidden`)
  polar     : the same Waring polynomial in (u = x/|x|, r = |x|) with K mode logits, and
              hallucination logit c + a*(r - sqrt d) - beta*(top1 - top2 mode logit):
              "modes = argmax of K degree-p polynomials on sphere x radius, hallucination =
              within a radius-dependent margin of a mode boundary"
  mlp       : depth-`depth` SiLU MLP of width `hidden` -> universal approximator (ceiling)

Non-parametric (no training; the labeled seeds ARE the model):
  knn       : vote of the `k` nearest labeled seeds (Euclidean, seed space)
  kernel    : Gaussian-kernel vote, h = `bandwidth` x median nearest-neighbour distance
              among the labeled seeds (Nadaraya-Watson; the classic atlas vote)
  altered_knn : weighted kNN over MODE-labeled anchors only (concentric rings with radius-
              decaying weights, see gmm.ring_anchors_weighted); a seed whose neighbour vote is
              low-confidence (normalised entropy) is declared a hallucination. See AlteredKNN.

Only `mlp` is universal; everything else is a fixed-capacity family whose boundaries can
be described in closed form (or, for knn / kernel, are the Voronoi / kernel cells of the
labeled seeds).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

PARAMETRIC = ("linear", "margin", "radial", "quadratic", "poly", "polar", "mlp")
NONPARAMETRIC = ("knn", "kernel", "altered_knn")
ARCHS = PARAMETRIC + NONPARAMETRIC


class NonParametric(nn.Module):
    """knn / kernel vote over stored labeled seeds; forward returns log class-frequencies."""
    def __init__(self, arch, K, k=10, bandwidth=1.0):
        super().__init__()
        self.arch, self.K, self.k, self.bandwidth = arch, K, int(k), float(bandwidth)
        self.register_buffer("X", torch.zeros(0))
        self.register_buffer("Y", torch.zeros(0))
        self.h = None

    @torch.no_grad()
    def fit(self, X, y):
        self.X = X
        self.Y = nn.functional.one_hot(y + 1, self.K + 1).float()
        if self.arch == "kernel":
            D = torch.cdist(X[:4096], X)
            D.fill_diagonal_(float("inf")) if D.shape[0] == D.shape[1] else None
            nn_dist = D.min(1).values
            self.h = self.bandwidth * float(nn_dist[torch.isfinite(nn_dist)].median())
        return self

    @torch.no_grad()
    def forward(self, x):
        chunk = max(16, int(2e8 // max(self.X.shape[0], 1)))     # ~800 MB of distances per block
        out = []
        for s in range(0, x.shape[0], chunk):
            D = torch.cdist(x[s:s + chunk], self.X)
            if self.arch == "knn":
                idx = D.topk(min(self.k, D.shape[1]), dim=1, largest=False).indices
                votes = self.Y[idx].mean(1)
            else:
                w = torch.exp(-(D ** 2) / (2 * self.h ** 2))
                votes = (w @ self.Y) / w.sum(1, keepdim=True).clamp_min(1e-30)
            out.append(torch.log(votes + 1e-9))
        return torch.cat(out)

    def describe(self):
        d = {"arch": self.arch, "n_stored": int(self.X.shape[0])}
        if self.arch == "knn":
            d["k"] = self.k
        else:
            d["bandwidth_rel"] = self.bandwidth; d["h"] = self.h
        return d


class AlteredKNN(nn.Module):
    """Weighted kNN over mode-labeled anchors; hallucination = low confidence.

    Anchors carry a mode label and a weight (no hallucination class). For a query seed the
    k nearest anchors vote, s_k = sum_i w_i [y_i = k] / sum_i w_i; p = softmax(s / temperature)
    (temperature <= 0: p = s). Confidence is 1 - H(p) / log K (`confidence: entropy`) or max p
    (`confidence: max`); the seed is a hallucination iff confidence < threshold, else argmax p.
    """
    def __init__(self, K, k=10, temperature=0.1, threshold=0.5, confidence="entropy"):
        super().__init__()
        self.arch, self.K, self.k = "altered_knn", K, int(k)
        self.temperature, self.confidence = float(temperature), confidence
        self.threshold = None if threshold == "auto" else float(threshold)   # None until calibrate()
        self.register_buffer("X", torch.zeros(0)); self.register_buffer("Y", torch.zeros(0))
        self.register_buffer("W", torch.zeros(0))

    @torch.no_grad()
    def fit(self, X, y, w=None):
        keep = y >= 0                                  # ignore any hallucination-labeled points
        X, y = X[keep], y[keep]
        w = torch.ones(X.shape[0], device=X.device) if w is None else w[keep].float()
        self.X, self.W = X, w
        self.Y = nn.functional.one_hot(y, self.K).float()
        return self

    @torch.no_grad()
    def probs(self, x):
        chunk = max(16, int(2e8 // max(self.X.shape[0], 1)))
        out = []
        for s in range(0, x.shape[0], chunk):
            D = torch.cdist(x[s:s + chunk], self.X)
            idx = D.topk(min(self.k, D.shape[1]), dim=1, largest=False).indices
            wv = self.W[idx]                                       # (n, k)
            votes = (wv[:, :, None] * self.Y[idx]).sum(1) / wv.sum(1, keepdim=True).clamp_min(1e-12)
            out.append(torch.softmax(votes / self.temperature, 1) if self.temperature > 0 else votes)
        return torch.cat(out)

    @torch.no_grad()
    def conf(self, x):
        p = self.probs(x)
        if self.confidence == "max":
            return p, p.max(1).values
        H = -(p * torch.log(p.clamp_min(1e-12))).sum(1)
        return p, 1 - H / math.log(max(self.K, 2))

    @torch.no_grad()
    def calibrate(self, X_cal, y_cal):
        """threshold = the confidence cut maximising hallucination F1 on (X_cal, y_cal)."""
        _, c = self.conf(X_cal)
        hall = y_cal == -1
        best, best_f1 = 0.0, -1.0
        for th in torch.quantile(c, torch.linspace(0.005, 0.5, 100, device=c.device)).tolist():
            pred = c < th
            tp = int((pred & hall).sum()); fp = int((pred & ~hall).sum()); fn = int((~pred & hall).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best, best_f1 = th, f1
        self.threshold, self.cal_f1 = best, best_f1
        return self

    @torch.no_grad()
    def forward(self, x):
        if self.threshold is None:
            raise RuntimeError("altered_knn with threshold=auto needs calibrate(X_cal, y_cal) before predicting")
        p, conf = self.conf(x)
        hall = conf < self.threshold
        big = 30.0
        logits = torch.log(p.clamp_min(1e-12))
        zh = torch.where(hall, torch.full_like(conf, big), torch.full_like(conf, -big))
        return torch.cat([zh[:, None], logits], 1)

    def describe(self):
        return {"arch": self.arch, "n_stored": int(self.X.shape[0]), "k": self.k,
                "temperature": self.temperature, "threshold": self.threshold, "confidence": self.confidence,
                **({"cal_hall_f1": self.cal_f1} if hasattr(self, "cal_f1") else {})}


class FateNet(nn.Module):
    def __init__(self, d, K, arch="mlp", hidden=512, depth=4, degree=3):
        super().__init__()
        if arch not in PARAMETRIC:
            raise ValueError(f"unknown parametric fate arch {arch!r}; choose from {PARAMETRIC}")
        self.arch, self.d, self.K, self.degree = arch, d, K, degree
        h = hidden
        if arch == "linear":
            self.net = nn.Linear(d, K + 1)
        elif arch == "margin":
            self.net = nn.Linear(d, K)
            self.c = nn.Parameter(torch.zeros(()))
            self.log_beta = nn.Parameter(torch.zeros(()))
        elif arch == "radial":
            self.net = nn.Linear(d + 1, K + 1)
        elif arch == "quadratic":
            self.register_buffer("iu", torch.triu_indices(d, d))
            self.net = nn.Linear(d + self.iu.shape[1], K + 1)
        elif arch == "poly":
            self.proj = nn.Linear(d, h)
            self.net = nn.Linear(h * degree, K + 1)
        elif arch == "polar":
            self.proj = nn.Linear(d + 1, h)
            self.net = nn.Linear(h * degree, K)
            self.c = nn.Parameter(torch.zeros(()))
            self.a = nn.Parameter(torch.zeros(()))
            self.log_beta = nn.Parameter(torch.zeros(()))
        else:  # mlp
            layers = [nn.Linear(d, h), nn.SiLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(h, h), nn.SiLU()]
            layers += [nn.Linear(h, K + 1)]          # class 0 = hallucination, class k+1 = mode k
            self.net = nn.Sequential(*layers)

    def _powers(self, z):
        return torch.cat([z ** q for q in range(1, self.degree + 1)], 1)

    def _with_margin(self, zm, extra=0.0):
        top = zm.topk(2, dim=1).values
        zh = self.c + extra - self.log_beta.exp() * (top[:, 0] - top[:, 1])
        return torch.cat([zh[:, None], zm], 1)

    def forward(self, x):
        a = self.arch
        if a == "margin":
            return self._with_margin(self.net(x))
        if a == "radial":
            return self.net(torch.cat([x, (x * x).sum(1, keepdim=True)], 1))
        if a == "quadratic":
            q = x[:, self.iu[0]] * x[:, self.iu[1]]
            return self.net(torch.cat([x, q], 1))
        if a == "poly":
            z = self.proj(x) / math.sqrt(self.d)
            return self.net(self._powers(z))
        if a == "polar":
            mu = math.sqrt(self.d)
            r = x.norm(dim=1, keepdim=True)
            rs = r - mu
            z = self.proj(torch.cat([x / r * math.sqrt(mu), rs], 1)) / math.sqrt(mu + 1)
            return self._with_margin(self.net(self._powers(z)), extra=self.a * rs[:, 0])
        return self.net(x)

    def describe(self) -> dict:
        """Readable summary of the fitted hallucination rule (margin / polar only)."""
        out = {"arch": self.arch, "n_params": sum(p.numel() for p in self.parameters())}
        if hasattr(self, "log_beta"):
            out["beta"] = float(self.log_beta.exp())
            out["c"] = float(self.c)
        if hasattr(self, "a"):
            out["a"] = float(self.a)
        return out


def train_fate_classifier(X, y, K, arch="mlp", hidden=512, depth=4, degree=3, epochs=30,
                          batch=2048, lr=3e-3, weight_decay=1e-4, min_steps=0, k=10, bandwidth=1.0,
                          temperature=0.1, threshold=0.5, confidence="entropy", w=None,
                          seed=0, device=None):
    """Fit one classifier on seeds X (n,d) with fate labels y in {-1, 0..K-1}.

    Parametric archs run `epochs` passes of mini-batches of size `batch`, but at least
    `min_steps` gradient steps, so small training sets (a few thousand ring anchors) converge.
    Non-parametric archs (knn, kernel) just store the labeled seeds.
    """
    if arch == "altered_knn":
        return AlteredKNN(K, k, temperature, threshold, confidence).to(device).fit(X, y, w)   # calibrate() if threshold=auto
    if arch in NONPARAMETRIC:
        return NonParametric(arch, K, k, bandwidth).to(device).fit(X, y)
    if arch not in ARCHS:
        raise ValueError(f"unknown fate arch {arch!r}; choose from {ARCHS}")
    torch.manual_seed(seed)
    net = FateNet(X.shape[1], K, arch, hidden, depth, degree).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = X.shape[0]
    batch = min(batch, n)
    per_epoch = n // batch
    epochs = max(epochs, -(-min_steps // per_epoch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * per_epoch)
    yl = y + 1
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n - batch + 1, batch):
            i = perm[s:s + batch]
            loss = nn.functional.cross_entropy(net(X[i]), yl[i])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net


def train_ensemble(X, y, K, spec: dict, device=None, w=None):
    """Train `spec['ensemble']` classifiers from one spec (see conf/classifier); w = anchor weights."""
    kw = dict(arch=spec["arch"], hidden=spec["hidden"], depth=spec["depth"], degree=spec["degree"],
              epochs=spec["epochs"], batch=spec["batch"], lr=spec["lr"],
              weight_decay=spec["weight_decay"], min_steps=spec.get("min_steps", 0),
              k=spec.get("k", 10), bandwidth=spec.get("bandwidth", 1.0),
              temperature=spec.get("temperature", 0.1), threshold=spec.get("threshold", 0.5),
              confidence=spec.get("confidence", "entropy"), device=device)
    if spec["arch"] == "altered_knn":
        kw["w"] = w
    n_ens = 1 if spec["arch"] in NONPARAMETRIC else spec["ensemble"]   # deterministic -> no ensemble
    return [train_fate_classifier(X, y, K, seed=e, **kw) for e in range(n_ens)]


@torch.no_grad()
def predict_fate(nets, X, chunk=100000):
    """Ensemble prediction (summed log-prob) -> labels in {-1, 0..K-1}."""
    out = []
    for s in range(0, X.shape[0], chunk):
        xs = X[s:s + chunk]
        lp = sum(torch.log_softmax(net(xs), 1) for net in nets)
        out.append(lp.argmax(1) - 1)
    return torch.cat(out)


METRICS = ("full_acc", "mode_acc", "mode_f1", "hall_prec", "hall_rec", "hall_f1", "balanced")


def _f1(pred_pos, gt_pos):
    tp = int((pred_pos & gt_pos).sum()); fp = int((pred_pos & ~gt_pos).sum()); fn = int((~pred_pos & gt_pos).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def fate_metrics(pred, gt) -> dict:
    """Scores of predicted vs ground-truth fates over held-out seeds.

    full_acc  : fraction of seeds whose fate (mode k or hallucination) is predicted exactly
    mode_acc  : same, restricted to seeds that truly reach a mode
    mode_f1   : mode-basin F1 -- macro average over modes k of the one-vs-rest F1 of
                "predicted k" vs "truly k" (all seeds count, so a hallucinating seed predicted
                as mode k is a false positive for k)
    hall_*    : precision / recall / F1 of the hallucination class (one-vs-rest)
    balanced  : (mode_acc + hall_f1) / 2
    """
    m_cls, m_hall = gt >= 0, gt == -1
    full = float((pred == gt).float().mean())
    mode = float((pred[m_cls] == gt[m_cls]).float().mean()) if m_cls.any() else float("nan")
    prec, rec, f1 = _f1(pred == -1, m_hall)
    modes = torch.unique(gt[m_cls]).tolist()
    mode_f1 = float(sum(_f1(pred == k, gt == k)[2] for k in modes) / max(len(modes), 1))
    return {"full_acc": full, "mode_acc": mode, "mode_f1": mode_f1, "hall_prec": prec,
            "hall_rec": rec, "hall_f1": f1, "balanced": 0.5 * (mode + f1) if mode == mode else float("nan")}
