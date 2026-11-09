"""Held-out evaluation of a submitted edit on LIBERO-Recovery (scenes never shown to the editor).

usage: python eval_recover.py --cap 25 --out <dir> [--orig-from <dir>] --edit name=<dir> ...
Fixed stratified sample of held-out scenes (up to --cap per suite x level, rng 20261104), both gripper starts, seed 7;
original once, each edit once (paired). Reports success by level/suite, re-grasp-loop failures and exact McNemar p.
"""
import argparse, collections, json, math, os, random, shutil, sys

F = "/mnt/SSD_16TB/jusheng/wsp_recover"
sys.path.insert(0, F)
import client as C  # noqa: E402


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orig-from", default=None)
    ap.add_argument("--edit", action="append", default=[])
    a = ap.parse_args()
    H = json.load(open(F + "/wsp/heldout_scenes.json"))
    E = {}
    for l in open(F + "/wsp/all_fixed.jsonl"):
        e = json.loads(l); E[(e["scene_id"], e["grip"])] = e
    cells = collections.defaultdict(list)
    for v in sorted(H, key=lambda v: v["scene_id"]):
        cells[(v["suite"], v["level"])].append(v)
    rng = random.Random(20261104)
    run = []
    for key in sorted(cells):
        pool = cells[key]
        for v in (pool if len(pool) <= a.cap else rng.sample(pool, a.cap)):
            for g in ("open", "close"):
                e = dict(E[(v["scene_id"], g)]); e["seed"] = 7; run.append(e)
    socks = [s for s in C.list_socks(F + "/run/sock/*.sock") if os.path.exists(s[:-5] + ".ready")]
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(run)} held-out episodes; {len(socks)} servers", flush=True)
    if a.orig_from and os.path.exists(a.orig_from + "/original_results.json"):
        orig = json.load(open(a.orig_from + "/original_results.json"))
    else:
        orig = C.rollout(run, a.out + "/original", socks=socks, arms=["orig"], log=lambda *x: None)
    json.dump(orig, open(a.out + "/original_results.json", "w"))
    res = {"original": orig}
    for spec in a.edit:
        name, d = spec.split("=", 1)
        shm = "/dev/shm/wsp_eval_" + name
        shutil.rmtree(shm, ignore_errors=True); shutil.copytree(d, shm)
        res[name] = C.rollout(run, a.out + "/" + name, socks=socks, weights_override=shm, arms=["edit"], log=lambda *x: None)
    key = lambda r: (r["scene_id"], r["grip"])
    ok = {m: {key(r): r for r in rs if r.get("termination") != "error"} for m, rs in res.items()}
    common = set.intersection(*(set(v) for v in ok.values()))
    info = {key(e): e for e in run}
    groups = [("ALL", lambda e: True)] + [(f"level {l}", (lambda l: lambda e: e["level"] == l)(l)) for l in ("L1", "L2", "L3", "L4")] \
        + [(s, (lambda s: lambda e: e["suite"] == s)(s)) for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10")]
    lines = ["| group | n | original | " + " | ".join(f"{m} (diff, +/-, McNemar p)" for m in res if m != "original") + " | loops orig -> edit |",
             "|---" * (4 + len(res) - 1) + "|"]
    summary = {}
    for gname, f in groups:
        ks = [k for k in common if f(info[k])]
        if not ks:
            continue
        o = ok["original"]
        row = [gname, str(len(ks)), f"{100 * sum(o[k]['success'] for k in ks) / len(ks):.1f}"]
        loops = []
        for m in res:
            if m == "original":
                continue
            e = ok[m]
            b = sum(e[k]["success"] and not o[k]["success"] for k in ks); c = sum(o[k]["success"] and not e[k]["success"] for k in ks)
            acc = 100 * sum(e[k]["success"] for k in ks) / len(ks)
            row.append(f"{acc:.1f} ({(b - c) * 100 / len(ks):+.1f}, +{b}/-{c}, p={mcnemar(b, c):.3g})")
            lp = lambda d: sum((not d[k]["success"]) and (d[k].get("grip_flips") or 0) >= 6 for k in ks)
            loops.append(f"{lp(o)} -> {lp(e)}")
            summary.setdefault(m, {})[gname] = {"acc": acc, "gain": b, "loss": c, "p": mcnemar(b, c), "n": len(ks)}
        summary.setdefault("original", {})[gname] = {"acc": 100 * sum(o[k]["success"] for k in ks) / len(ks)}
        lines.append("| " + " | ".join(row) + " | " + "; ".join(loops) + " |")
    md = "\n".join(lines)
    open(a.out + "/summary.md", "w").write(md + "\n")
    json.dump(summary, open(a.out + "/summary.json", "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
