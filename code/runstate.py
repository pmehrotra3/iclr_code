"""runstate.py — one named run = one folder, built up by as many invocations as you like.

`run_id` is a NAME you pick (`run_id=abc123`), not a timestamp. Everything the run produces
lives under output/<run_id>/: checkpoints/<process>/<variant>/ (learned samplers, gt caches,
manifest), <process>/<variant>/T<T>/ (per-cell result files, results.json, figures) and logs/
(scripts/main.sh). Every stage skips work whose checkpoint / cell file already exists, so
re-running with the same run_id resumes where it stopped, and a sweep split over several
small invocations (other d, more seeds, more anchor budgets, another T_true, another predictor,
the other data.weighted variant) ends up with exactly the files one big invocation would have
written.

That only holds if every invocation uses the same experiment settings. So the first invocation
records them in output/<run_id>/run.json, and every later one is checked against it:

  may differ between invocations (they select WHICH cells / rows to compute):
      stages, device, seed, n_seeds, sweep.* (d, K, anchors), process name + T_true,
      data.weighted (the variant), the predictor LIST (new predictors are added), eval.part,
      eval.force, *.force_retrain, train.graph_streams
  must match (they change WHAT a cell computes):
      everything else -- data contract, train recipe, process schedule, eval sizes, anchors,
      and each predictor's own hyper-parameters

A mismatch stops the invocation with the list of differences. Pick a new run_id for a
different experiment, or pass strict_run=false to override on purpose.
"""
from __future__ import annotations
import fcntl
import json
import os
import time

from omegaconf import OmegaConf

# top-level keys that only select work, never change a result
_SELECT = {"stages", "run_id", "device", "seed", "n_seeds", "sweep", "paths", "strict_run", "hydra"}
# keys dropped wherever they appear (paths and per-invocation switches)
_DROP_ANY = {"root", "force_retrain", "force", "graph_streams", "part"}


def run_dir(cfg) -> str:
    return os.path.join(cfg.paths.output, str(cfg.run_id))


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in _DROP_ANY}
    return x


def identity(cfg) -> dict:
    """The settings that must agree across every invocation of one run (see module doc)."""
    c = OmegaConf.to_container(cfg, resolve=True)
    proc = _strip(dict(c.get("process", {})))
    proc.pop("T_true", None)
    pred = dict(c.get("classifier", {}))
    shared = pred.get("_shared", {}) or {}
    models = pred.get("models", {}) or {}
    ident = {k: _strip(v) for k, v in c.items() if k not in _SELECT | {"process", "classifier"}}
    ident.get("data", {}).pop("weighted", None)          # the variant: selects, never changes
    return {"config": ident,
            "process": {str(proc.get("name")): proc},
            "predictors": {n: {**shared, **(m or {})} for n, m in models.items()}}


def _diff(old, new, path=""):
    """Leaves present in both with different values (keys only one side has are not a diff)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in old.keys() & new.keys():
            out += _diff(old[k], new[k], f"{path}.{k}" if path else str(k))
        return out
    return [] if old == new else [(path, old, new)]


def _merge(old, new):
    if isinstance(old, dict) and isinstance(new, dict):
        out = dict(old)
        for k, v in new.items():
            out[k] = _merge(old[k], v) if k in old else v
        return out
    return old


def check_and_record(cfg, overrides=()) -> str:
    """Check this invocation against output/<run_id>/run.json (creating it on the first one),
    add any new process / predictor to it, and log the invocation. Returns the run folder."""
    rd = run_dir(cfg)
    os.makedirs(rd, exist_ok=True)
    new = identity(cfg)
    with open(os.path.join(rd, "run.json"), "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)                  # parallel jobs of scripts/main.sh
        f.seek(0)
        txt = f.read()
        old = json.loads(txt) if txt.strip() else None
        if old is not None:
            diffs = [(("config." + p) if p else "config", a, b) for p, a, b in _diff(old["config"], new["config"])]
            diffs += [("process." + p, a, b) for p, a, b in _diff(old["process"], new["process"])]
            diffs += [("predictor." + p, a, b) for p, a, b in _diff(old["predictors"], new["predictors"])]
            if diffs and bool(cfg.get("strict_run", True)):
                fcntl.flock(f, fcntl.LOCK_UN)
                lines = "\n".join(f"    {p}: run has {a!r}, this invocation {b!r}" for p, a, b in diffs)
                raise ValueError(
                    f"run '{cfg.run_id}' was made with different settings ({rd}/run.json):\n{lines}\n"
                    f"  -> use a new run_id for a different experiment, or strict_run=false to mix anyway")
            merged = {**_merge(old, new), "created": old.get("created")}
        else:
            merged = {**new, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        merged["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.seek(0); f.truncate()
        json.dump(merged, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)

    # the invocation log: one line per invocation, plus its resolved config
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    stages = "_".join(cfg.stages)
    os.makedirs(os.path.join(rd, "invocations"), exist_ok=True)
    with open(os.path.join(rd, "invocations", f"{stamp}_{os.getpid()}_{stages}.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(os.path.join(rd, "history.log"), "a") as f:
        f.write(f"{stamp}  pid={os.getpid()}  stages=[{','.join(cfg.stages)}]  {' '.join(overrides)}\n")
    return rd
