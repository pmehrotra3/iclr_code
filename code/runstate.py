"""runstate.py — a run is a name, and every invocation under that name must agree on settings.

`run_id` (e.g. abc123) names a run. Its results live in output/<run_id>/ and its models in
checkpoints/. Every stage skips work that is already done, so a sweep can be built up over many
invocations (more d, more seeds, another anchor budget, another T_true, the weighted variant, a
new predictor) and ends with the same files as one big invocation.

That only holds if the invocations compute each cell the same way. The first one writes its
settings to output/<run_id>/run.json; later ones are compared against it and refused if
anything that changes a result differs (the training recipe, the eval size, a predictor's
hyper-parameters, ...). Settings that only choose WHICH cells to compute may differ: see
_SELECT and _DROP below. Pass strict_run=false to mix settings on purpose.
"""
from __future__ import annotations
import fcntl
import json
import os
import time

from omegaconf import OmegaConf

# top-level keys that only choose what to compute (plus data.weighted, the variant)
_SELECT = {"stages", "run_id", "device", "seed", "n_seeds", "sweep", "paths", "strict_run", "hydra"}
# per-invocation switches, ignored wherever they appear
_DROP = {"root", "force_retrain", "force", "graph_streams", "part"}


def run_dir(cfg):
    return os.path.join(cfg.paths.output, str(cfg.run_id))


def _strip(x):
    return {k: _strip(v) for k, v in x.items() if k not in _DROP} if isinstance(x, dict) else x


def identity(cfg):
    """The settings every invocation of a run must share, split into config / process /
    predictors so a new process or predictor can be added to a run later."""
    c = OmegaConf.to_container(cfg, resolve=True)
    proc = _strip(c["process"])
    proc.pop("T_true", None)
    shared = c["classifier"].get("_shared") or {}
    config = {k: _strip(v) for k, v in c.items() if k not in _SELECT | {"process", "classifier"}}
    config["data"].pop("weighted", None)
    return {"config": config,
            "process": {proc["name"]: proc},
            "predictors": {n: {**shared, **(m or {})} for n, m in c["classifier"]["models"].items()}}


def _diff(old, new, path=""):
    """[(path, old, new)] for values both sides have that differ; a key only one side has
    (a new predictor, a setting added to the config later) is not a difference."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [x for k in old.keys() & new.keys() for x in _diff(old[k], new[k], f"{path}.{k}" if path else k)]
    return [] if old == new else [(path, old, new)]


def _merge(old, new):
    """`old` plus whatever only `new` has (new processes / predictors)."""
    if isinstance(old, dict) and isinstance(new, dict):
        return {**{k: _merge(v, new[k]) if k in new else v for k, v in old.items()},
                **{k: v for k, v in new.items() if k not in old}}
    return old


def settings_diff(old, new):
    """Differences between two run.json records, as (section.path, old, new)."""
    return [(f"{name}.{p}" if p else name, a, b)
            for name, key in (("config", "config"), ("process", "process"), ("predictor", "predictors"))
            for p, a, b in _diff(old[key], new[key])]


def check_and_record(cfg, overrides=()):
    """Compare this invocation with output/<run_id>/run.json (the first invocation creates it),
    record any new process / predictor, and log the invocation. Returns the run folder."""
    rd = run_dir(cfg)
    os.makedirs(rd, exist_ok=True)
    new = identity(cfg)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(rd, "run.json"), "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)                  # scripts/main.sh starts jobs in parallel
        f.seek(0)
        text = f.read()
        if text.strip():
            old = json.loads(text)
            diffs = settings_diff(old, new)
            if diffs and cfg.strict_run:
                fcntl.flock(f, fcntl.LOCK_UN)
                lines = "\n".join(f"    {p}: run has {a!r}, this invocation {b!r}" for p, a, b in diffs)
                raise ValueError(f"run '{cfg.run_id}' was made with different settings ({rd}/run.json):\n"
                                 f"{lines}\n  -> use a new run_id, or strict_run=false to mix on purpose")
            record = {**_merge(old, new), "created": old.get("created"), "updated": now}
        else:
            record = {**new, "created": now, "updated": now}
        f.seek(0); f.truncate()
        json.dump(record, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)

    # keep the full resolved config of every invocation, and a one-line history
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(os.path.join(rd, "invocations"), exist_ok=True)
    with open(os.path.join(rd, "invocations", f"{stamp}_{os.getpid()}_{'_'.join(cfg.stages)}.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(os.path.join(rd, "history.log"), "a") as f:
        f.write(f"{stamp}  pid={os.getpid()}  stages=[{','.join(cfg.stages)}]  {' '.join(overrides)}\n")
    return rd
