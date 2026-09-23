"""combine.py — fold a run built on ANOTHER machine into this machine's run of the same name.

A run can be built on several machines at once (e.g. d <= 256 here, d = 512 elsewhere) with the
same run_id and settings. Copy the other machine's checkpoints/ and output/<run_id>/ into one
folder, keeping those two names, then:

    rsync -a OTHER:iclr_code/checkpoints OTHER:iclr_code/output /tmp/other/
    python code/combine.py /tmp/other --dry-run      # report only
    python code/combine.py /tmp/other                # merge into ./checkpoints and ./output/<run_id>

Steps:
  1. settings   the two run.json files must agree (the check every invocation does; --force to
                combine anyway); processes / predictors new on the other side are added
  2. conflicts  a (process, variant, d, K, seed) model trained on BOTH machines with different
                weights: this side's model is kept, and nothing the other side derived from its
                own model (gt cache, cell rows) comes over, so results never mix two models
  3. files      every file this side lacks is copied (models, gt caches, cell files, logs,
                invocations); cells/*.json of the same cell get the union of their rows (this
                side wins a row both have); manifest.json and history.log get the union
  4. rebuild    results.json / csv and the figures of every <process>/<variant>/T<T>/, from all
                cell files (--no-rebuild skips it). The other side's results.* / *.png are
                never copied.
"""
from __future__ import annotations
import argparse
import filecmp
import fcntl
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runstate  # noqa: E402

REBUILT = ("results.json", "results.csv", "results_per_seed.csv", ".lock")


def _copy(src, dst, dry):
    if not dry:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.{os.getpid()}.tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)                                  # atomic


def _write_json(path, obj, dry):
    if not dry:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)


def _relocate(manifest, dst_root):
    """Point the other machine's absolute checkpoint paths at this machine's checkpoints/."""
    tag = os.sep + "checkpoints" + os.sep
    for e in manifest.values():
        p = e.get("path") if isinstance(e, dict) else None
        if p and tag in p:
            e["path"] = os.path.join(dst_root, "checkpoints", p.split(tag, 1)[1])
    return manifest


def _same_model(a, b):
    """Identical bytes, or identical weights (metadata such as 'created' may differ)."""
    if filecmp.cmp(a, b, shallow=False):
        return True
    import torch
    sa = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    sb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def check_settings(src_run, dst_run, force, dry):
    """Step 1: compare the two run.json files and write the merged one."""
    ps, pd = os.path.join(src_run, "run.json"), os.path.join(dst_run, "run.json")
    if not os.path.exists(ps):
        raise FileNotFoundError(f"{ps} missing: expected <src>/output/<run_id>/run.json")
    new = json.load(open(ps))
    if not os.path.exists(pd):
        print("  run.json: none here yet, taking the other side's")
        _copy(ps, pd, dry)
        return
    with open(pd, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        old = json.load(f)
        diffs = runstate.settings_diff(old, new)
        if diffs:
            lines = "\n".join(f"    {p}: here {a!r}, other side {b!r}" for p, a, b in diffs)
            if not force:
                raise ValueError(f"the two runs were made with different settings:\n{lines}\n"
                                 f"  -> they are different experiments; --force to combine anyway")
            print(f"  run.json: settings differ, combining anyway (--force):\n{lines}")
        merged = {**runstate._merge(old, new),
                  "created": min(old.get("created") or "~", new.get("created") or "~"),
                  "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        if not dry:
            f.seek(0); f.truncate()
            json.dump(merged, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)
    print("  run.json: " + ("merged" if diffs else "same settings"))


def _conflicts(src_root, dst_root):
    """Step 2: {(process, variant, 'd<d>_K<K>_s<seed>')} trained on both sides differently."""
    out = set()
    for s in glob.glob(os.path.join(src_root, "checkpoints", "*", "*", "checkpoints", "model_*.pt")):
        rel = os.path.relpath(s, src_root)
        d = os.path.join(dst_root, rel)
        if os.path.exists(d) and not _same_model(s, d):
            proc, var = rel.split(os.sep)[1:3]
            out.add((proc, var, os.path.basename(s)[len("model_"):-3]))
    return out


def combine(src_root, dst_root, run_id, force=False, dry=False):
    src_root, dst_root = os.path.abspath(src_root), os.path.abspath(dst_root)
    if src_root == dst_root:
        raise ValueError("source and destination are the same folder")
    src_run, dst_run = (os.path.join(r, "output", run_id) for r in (src_root, dst_root))
    print(f"combine {src_root} (run {run_id})\n     -> {dst_root}" + ("   [dry run]" if dry else ""))
    check_settings(src_run, dst_run, force, dry)
    conflicts = _conflicts(src_root, dst_root)
    for c in sorted(conflicts):
        print(f"  CONFLICT {c[0]}/{c[1]} {c[2]}: trained on both machines with different weights "
              f"-> keeping this side's model and everything derived from it")

    def conflicted(parts):
        name = os.path.splitext(parts[-1])[0].replace("model_", "")
        if parts[0] == "checkpoints" and len(parts) >= 5:         # checkpoints/<proc>/<var>/...
            return (parts[1], parts[2], name) in conflicts
        if parts[-2] == "cells":                                  # output/<run>/<proc>/<var>/T/cells/
            return (parts[2], parts[3], name) in conflicts
        return False

    stats = dict.fromkeys(("copied", "same", "cells_merged", "skipped_conflict", "kept_ours"), 0)
    for tree in (os.path.join(src_root, "checkpoints"), src_run):
        for root, _, files in os.walk(tree):
            for fn in files:
                s = os.path.join(root, fn)
                rel = os.path.relpath(s, src_root)
                d = os.path.join(dst_root, rel)
                if fn in REBUILT or fn in ("run.json", "history.log") or fn.endswith((".png", ".tmp")):
                    continue
                if conflicted(rel.split(os.sep)):
                    stats["skipped_conflict"] += 1
                elif fn == "manifest.json":
                    ours = json.load(open(d)) if os.path.exists(d) else {}
                    _write_json(d, {**_relocate(json.load(open(s)), dst_root), **ours}, dry)
                elif not os.path.exists(d):
                    _copy(s, d, dry)
                    stats["copied"] += 1
                elif filecmp.cmp(s, d, shallow=False) or fn.startswith("model_"):
                    stats["same"] += 1                  # a model on both sides with the same weights
                elif rel.split(os.sep)[-2] == "cells":
                    here = json.load(open(d))["rows"]
                    have = {(int(r["n_per_mode"]), r["model"]) for r in here}
                    extra = [r for r in json.load(open(s))["rows"]
                             if (int(r["n_per_mode"]), r["model"]) not in have]
                    _write_json(d, {"rows": here + extra}, dry)   # evaluate.merge re-sorts them
                    stats["cells_merged" if extra else "same"] += 1
                else:
                    print(f"  kept this side's {rel} (the other side's copy differs)")
                    stats["kept_ours"] += 1

    hs, hd = os.path.join(src_run, "history.log"), os.path.join(dst_run, "history.log")
    if os.path.exists(hs) and not dry:
        lines = set(open(hd).read().splitlines()) if os.path.exists(hd) else set()
        lines |= set(open(hs).read().splitlines())
        with open(hd, "w") as f:
            f.write("\n".join(sorted(l for l in lines if l.strip())) + "\n")
            f.write(f"{time.strftime('%Y-%m-%d_%H-%M-%S')}  combine  from {src_root}\n")
    print("  files: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    return stats


def rebuild(dst_root, run_id):
    """Step 4: results.json / csv + figures of every <process>/<variant>/T<T>/ with cell files."""
    from hydra import compose, initialize_config_dir
    import evaluate
    import visualize
    conf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")
    run = os.path.join(dst_root, "output", run_id)
    for cells in sorted(glob.glob(os.path.join(run, "*", "*", "T*", "cells"))):
        proc, var, T = os.path.relpath(os.path.dirname(cells), run).split(os.sep)
        with initialize_config_dir(config_dir=conf, version_base=None):
            cfg = compose("config", overrides=[
                f"run_id={run_id}", f"paths.root={dst_root}", f"process={proc}",
                f"data.weighted={'true' if var == 'weighted' else 'false'}", f"process.T_true={T[1:]}"])
        evaluate.merge(cfg)
        try:
            visualize.run(cfg)
        except Exception as e:                        # figures are a convenience; results are not
            print(f"  [visualize {proc}/{var}/{T}] skipped: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="folder holding the other machine's checkpoints/ and output/<run_id>/")
    ap.add_argument("--run", help="run_id (default: the only run under <src>/output/)")
    ap.add_argument("--into", help="this repository's root (default: $ATLAS_ROOT or the cwd)")
    ap.add_argument("--force", action="store_true", help="combine even if the settings differ")
    ap.add_argument("--dry-run", action="store_true", help="report only; change nothing")
    ap.add_argument("--no-rebuild", action="store_true", help="do not rebuild results.json / figures")
    a = ap.parse_args()
    run_id = a.run
    if run_id is None:
        runs = [r for r in os.listdir(os.path.join(a.src, "output"))
                if os.path.exists(os.path.join(a.src, "output", r, "run.json"))]
        if len(runs) != 1:
            ap.error(f"pass --run: runs under {a.src}/output: {runs}")
        run_id = runs[0]
    dst = os.path.abspath(a.into or os.environ.get("ATLAS_ROOT", os.getcwd()))
    combine(a.src, dst, run_id, a.force, a.dry_run)
    if not (a.dry_run or a.no_rebuild):
        rebuild(dst, run_id)
    print("done.")


if __name__ == "__main__":
    main()
