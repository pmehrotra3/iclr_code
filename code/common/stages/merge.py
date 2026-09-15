"""stages/merge.py — combine partial results (scripts/run_parallel.sh splits the sweep by
d across GPUs into output/_parts/<process>/<part>/<process>/results[_tag].json) into one
results file, as if evaluate had run in a single process.
"""
from __future__ import annotations
import os
import glob
import json

from common import utils
from common.stages.evaluate import write_results


def run(cfg):
    pname = cfg.process
    out_dir = utils.process_dir(cfg.paths.output, pname)
    fname = os.path.basename(utils.results_path(cfg.paths.output, pname, cfg.eval.tag, "json"))
    parts = sorted(glob.glob(os.path.join(cfg.paths.output, "_parts", pname, "*", pname, fname)))
    if not parts:
        raise FileNotFoundError(f"no partial results under {cfg.paths.output}/_parts/{pname}/*/{pname}/{fname}")
    results = []
    for p in parts:
        with open(p) as f:
            blob = json.load(f)
        if blob.get("process") != pname or any("classifiers" not in r for r in blob["results"]):
            raise ValueError(f"{p} is not a results file of this pipeline (stale _parts?)")
        results += blob["results"]
    results.sort(key=lambda r: (r["d"], r["K"]))
    print(f"[merge:{pname}] {len(parts)} parts -> {len(results)} cells")
    return write_results(cfg, results, out_dir)
