"""common/nets.py — the network backbone shared by every process.

ScoreNet(x, t) -> R^d. DDIM reads its output as eps, flow matching as a velocity; the
process decides. Time enters through a sinusoidal embedding of an integer step index.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = math.log(10000) / (half - 1)
        f = torch.exp(torch.arange(half, device=t.device) * -f)
        a = t[:, None].float() * f[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)


class MLPBlock(nn.Module):
    def __init__(self, h, act):
        super().__init__()
        self.n = nn.LayerNorm(h)
        self.a = act
        self.f1 = nn.Linear(h, h)
        self.f2 = nn.Linear(h, h)

    def forward(self, x):
        z = self.n(x); z = self.a(z); z = self.f1(z); z = self.a(z); z = self.f2(z)
        return x + z


class ScoreNet(nn.Module):
    def __init__(self, d, h=256, nb=4, td=128):
        super().__init__()
        self.arch = {"h": h, "nb": nb, "td": td}
        act = nn.LeakyReLU(0.2)
        self.inp = nn.Linear(d, h)
        self.te = SinusoidalPosEmb(td)
        self.tm = nn.Sequential(nn.Linear(td, h), act)
        self.bl = nn.ModuleList([MLPBlock(h, act) for _ in range(nb)])
        self.out = nn.Sequential(nn.LayerNorm(h), act, nn.Linear(h, d))

    def forward(self, x, t):
        z = self.inp(x) + self.tm(self.te(t))
        for b in self.bl:
            z = b(z)
        return self.out(z)
