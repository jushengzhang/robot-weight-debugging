"""Held-out evaluation of submitted edits (never shown to the editor).

usage: python eval_heldout.py --per-cat 60 --target-n 200 --out <dir> [--edit name=<run>/work/edited ...]
A fixed stratified sample of held-out variants (seed 7, one episode each) is run once with the original weights and
once per edit (paired by variant+seed). Reports success per category with paired differences and exact McNemar p.
"""
import argparse, collections, json, math, os, random, shutil, sys

F = "/mnt/SSD2_16TB/jusheng/wsp/fastwam"
sys.path.insert(0, F)
import client as C  # noqa: E402


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def sample(per_cat, target, target_n):
    H = json.load(open(F + "/heldout_variants.json"))
    by = collections.defaultdict(list)
    for v in H:
        by[v["category"]].append(v)
    rng = random.Random(20261103)
    eps = []
    for cat, vs in sorted(by.items()):
        vs = sorted(vs, key=lambda v: (v["suite"], v["id"]))
        n = target_n if cat in target else per_cat
        for v in rng.sample(vs, min(n, len(vs))):
            eps.append({"suite": v["suite"], "variant_id": int(v["id"]), "seed": 7, "category": cat})
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=60)
    ap.add_argument("--target-n", type=int, default=200)
    ap.add_argument("--targets", default="Camera Viewpoints,Background Textures,Language Instructions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="strip", choices=["strip", "raw"])
    ap.add_argument("--orig-from", default=None, help="reuse original_results.json of an earlier eval with the same sample")
    ap.add_argument("--edit", action="append", default=[], help="name=dir")
    a = ap.parse_args()
    eps = sample(a.per_cat, set(a.targets.split(",")), a.target_n)
    cat = {(e["suite"], e["variant_id"]): e["category"] for e in eps}
    run = [{k: e[k] for k in ("suite", "variant_id", "seed")} for e in eps]
    socks = [s for s in C.list_socks() if os.path.exists(s[:-5] + ".ready")]
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(run)} held-out episodes; {len(socks)} servers", flush=True)
    if a.orig_from and os.path.exists(a.orig_from + "/original_results.json"):
        res = {"original": json.load(open(a.orig_from + "/original_results.json"))}
        assert {(r["suite"], r["variant_id"]) for r in res["original"]} == set(cat), "cached original sample differs"
    else:
        res = {"original": C.rollout(run, a.out + "/original", socks=socks, arms=["orig"], language=a.language, log=lambda *x: None)}
    json.dump(res["original"], open(a.out + "/original_results.json", "w"))
    for spec in a.edit:
        name, d = spec.split("=", 1)
        shm = "/dev/shm/wsp_eval_" + name  # servers re-read the override every episode; the run disk is slow
        shutil.rmtree(shm, ignore_errors=True); shutil.copytree(d, shm); d = shm
        res[name] = C.rollout(run, a.out + "/" + name, socks=socks, weights_override=d, arms=["edit"], language=a.language, log=lambda *x: None)
        print("done", name, flush=True)
    succ = {m: {(r["suite"], r["variant_id"]): bool(r["success"]) for r in rs} for m, rs in res.items()}
    cats = sorted(set(cat.values()))
    lines = ["| category | n | original | " + " | ".join(f"{m} (diff, McNemar p)" for m in res if m != "original") + " |",
             "|---" * (3 + len(res) - 1) + "|"]
    summary = {}
    for c in cats + ["ALL"]:
        keys = [k for k, v in cat.items() if c == "ALL" or v == c]
        o = succ["original"]
        row = [c, str(len(keys)), f"{100 * sum(o[k] for k in keys) / len(keys):.1f}"]
        for m in res:
            if m == "original":
                continue
            e = succ[m]
            b = sum(e[k] and not o[k] for k in keys); cc = sum(o[k] and not e[k] for k in keys)
            acc = 100 * sum(e[k] for k in keys) / len(keys)
            row.append(f"{acc:.1f} ({(b - cc) * 100 / len(keys):+.1f}, +{b}/-{cc}, p={mcnemar(b, cc):.3g})")
            summary.setdefault(m, {})[c] = {"acc": acc, "gain": b, "loss": cc, "p": mcnemar(b, cc)}
        summary.setdefault("original", {})[c] = {"acc": 100 * sum(o[k] for k in keys) / len(keys)}
        lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(lines)
    open(a.out + "/summary.md", "w").write(md + "\n")
    json.dump(summary, open(a.out + "/summary.json", "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
