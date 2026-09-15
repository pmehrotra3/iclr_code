"""common/utils.py — device selection, per-process paths, seeds."""
from __future__ import annotations
import os
import torch


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seeds(N: int, d: int, seed: int, device: torch.device) -> torch.Tensor:
    """N Gaussian seeds in R^d, reproducible from `seed` on any device."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(N, d, generator=g, device=device)


# ------------------------------------------------------------------ paths
# Every process writes under its own name so runs never collide:
#   checkpoints/<process>/model_d{d}_K{K}_T{T}.pt     (train)
#   output/<process>/results[<tag>].{json,csv}         (evaluate / merge)
#   visualization/<process>/*.{pdf,png,tex}             (visualize / seedmap)
def process_dir(base: str, process: str) -> str:
    return os.path.join(base, process)


def ckpt_path(ckpt_dir: str, process: str, d: int, K: int, T: int) -> str:
    return os.path.join(process_dir(ckpt_dir, process), f"model_d{d}_K{K}_T{T}.pt")


def results_path(output_dir: str, process: str, tag: str = "", ext: str = "json") -> str:
    tag = f"_{tag}" if tag else ""
    return os.path.join(process_dir(output_dir, process), f"results{tag}.{ext}")
