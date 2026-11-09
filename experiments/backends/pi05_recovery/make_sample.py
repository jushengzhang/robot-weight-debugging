#!/usr/bin/env python
"""Stratified LIBERO-Recovery sample for pi05_libero (trained on spatial/object/goal/10; libero_90 excluded).
  python make_sample.py --cap 15 --grips open,close --out samples/rec220x2.jsonl
Per (suite, level) cell: up to --cap scenes (random.Random(seed)); each scene -> one episode per grip
(trial index = position in --grips). instruction = annotation 'task' (= LIBERO task language)."""
import argparse, collections, glob, json, os, random
from pathlib import Path
R = Path(__file__).resolve().parent
D = R / "data/FailureScenes"
ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, default=15)
ap.add_argument("--grips", default="open,close")
ap.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pert-mode", default="xy")
ap.add_argument("--out", required=True)
a = ap.parse_args()
cells = collections.defaultdict(list)
for suite in a.suites.split(","):
    for f in sorted(glob.glob(str(D / "annotations/scenes" / suite / "*.json"))):
        d = json.load(open(f))
        bddl = D / d["bddl_file"].removeprefix("benchmark_failure/")
        npy = D / d["sim_state_file"].removeprefix("benchmark_failure/")
        if not (bddl.exists() and npy.exists()):
            continue
        d["_bddl"], d["_npy"] = str(bddl), str(npy)
        cells[(suite, d["level"])].append(d)
rng = random.Random(a.seed)
eps = []
for key in sorted(cells):
    pool = cells[key]
    pick = pool if len(pool) <= a.cap else rng.sample(pool, a.cap)
    print(key, len(pool), "->", len(pick))
    for d in sorted(pick, key=lambda x: x["scene_id"]):
        for k, g in enumerate(a.grips.split(",")):
            eps.append({"suite": d["source"], "variant_id": f"{d['scene_id']}#{g}", "seed": 7, "init_idx": k,
                        "bddl": d["_bddl"], "sim_state": d["_npy"], "grip": g, "instruction": d["task"],
                        "scene_id": d["scene_id"], "level": d["level"], "stage": d.get("stage"),
                        "l4_injected": d["scene_id"].startswith("l4_injected"), "src_model": d.get("model"),
                        "pert_mode": a.pert_mode, "obj_noise_xy": 0.02, "settle": 20, "max_steps": 520})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w") as f:
    for e in eps:
        f.write(json.dumps(e) + "\n")
print(len(eps), "episodes,", len({e["scene_id"] for e in eps}), "scenes ->", a.out)
