"""Forward-pass tool for the FastWAM domain: maps the editor's ```forward requests onto the policy-server client."""
import collections, json, math, os, random, shutil, sys

F = "/mnt/SSD2_16TB/jusheng/wsp/fastwam"
BENCH = "/mnt/SSD2_16TB/jusheng/wsp/bench"
sys.path.insert(0, F)
import client as C  # noqa: E402

MAX_EP, MAX_EP_BOTH, MAX_PROBE = 256, 128, 512
LANG = os.environ.get("WSP_LANGUAGE")  # set (raw) only for the instruction-suffix domain; other runs always use strip
_PRAC = None


def _snapshot(ed, R, tag):
    """Copy WORK/edited to tmpfs: every server re-reads the override for each episode, and the run disk reads only
    a few MB/s under load (a 3 GB edit took minutes per episode). Older snapshots of the same run are removed."""
    root = "/dev/shm/wsp_fw_" + os.path.basename(os.path.abspath(R))
    os.makedirs(root, exist_ok=True)
    for old in os.listdir(root):
        shutil.rmtree(os.path.join(root, old), ignore_errors=True)
    dst = os.path.join(root, tag + "_weights")
    shutil.copytree(ed, dst)
    return dst


def ready_socks():
    return [s for s in C.list_socks() if os.path.exists(s[:-5] + ".ready")]


def practice():
    global _PRAC
    if _PRAC is None:
        _PRAC = {(v["suite"], int(v["id"])): v for v in json.load(open(BENCH + "/practice_variants.json"))}
    return _PRAC


def _episodes(req):
    P = practice()
    if "episodes" in req:
        eps = []
        for e in req["episodes"]:
            key = (e.get("suite"), int(e.get("id", e.get("variant_id", -1))))
            if key not in P:
                raise ValueError(f"{key} is not a practice variant (only BENCH/practice_variants.json may be run)")
            eps.append({"suite": key[0], "variant_id": key[1], "seed": int(e.get("seed", 7))})
        return eps
    s = req.get("sample")
    if not isinstance(s, dict) or "n" not in s:
        raise ValueError("give 'episodes' or 'sample' {category, suite?, difficulty?, n, seed}")
    pool = [v for v in P.values() if (not s.get("category") or v["category"] == s["category"])
            and (not s.get("suite") or v["suite"] == s["suite"])
            and (not s.get("difficulty") or v["difficulty"] in s["difficulty"])]
    if not pool:
        raise ValueError("no practice variant matches the sample filter")
    rng = random.Random(int(s.get("seed", 0)))
    pool = sorted(pool, key=lambda v: (v["suite"], v["id"]))
    pick = rng.sample(pool, min(int(s["n"]), len(pool)))
    return [{"suite": v["suite"], "variant_id": int(v["id"]), "seed": int(s.get("episode_seed", 7))} for v in pick]


def _rate(rs):
    return f"{100 * sum(r['success'] for r in rs) / len(rs):.1f}% ({sum(r['success'] for r in rs)}/{len(rs)})" if rs else "-"


def _table(rs, field):
    g = collections.defaultdict(list)
    for r in rs:
        g[str(r.get(field))].append(r)
    return {k: _rate(v) for k, v in sorted(g.items())}


def rollout(req, R, k):
    W = R + "/work"
    model = req.get("model", "original")
    arms = {"original": ["orig"], "edited": ["edit"], "both": ["orig", "edit"]}.get(model)
    if arms is None:
        raise ValueError("model must be original, edited or both")
    eps = _episodes(req)
    lim = MAX_EP_BOTH if model == "both" else MAX_EP
    if not eps or len(eps) > lim:
        raise ValueError(f"{len(eps)} episodes; allowed 1..{lim} for model={model}")
    ed = W + "/edited"
    override = None
    if "edit" in arms:
        if not (os.path.isdir(ed) and any(f.endswith(".npy") for f in os.listdir(ed))):
            raise ValueError("no edits saved in WORK/edited")
        override = _snapshot(ed, R, f"roll_{k}")  # snapshot so later edits don't race with the running pass
    rec = None
    if req.get("record"):
        r = req["record"]
        what = r.get("what", "attn_out"); what = what if isinstance(what, list) else [what]
        rec = {"action_blocks": list(r.get("action_layers", [])), "video_blocks": list(r.get("video_layers", [])), "what": what}
    lang = "strip"
    if LANG:
        lang = req.get("language", LANG)
        if lang not in ("raw", "strip"):
            raise ValueError("language must be raw or strip")
    out = W + f"/fwd/roll_{k}"
    res = C.rollout(eps, out, socks=ready_socks(), weights_override=override, arms=arms, record=rec,
                    save_frames=50 if req.get("save_frames") else 0, language=lang, log=lambda *a: None)
    errs = [r for r in res if r.get("termination") == "error"]
    if len(errs) > max(2, len(res) // 10):
        raise RuntimeError(f"{len(errs)} episodes errored: {errs[0].get('failure')}")
    if any(r.get("arm") == "edit" and r.get("restore_verified") is False for r in res):
        raise RuntimeError("weight restore verification failed")
    # editor-facing records
    acts_dir, frames_dir = W + f"/roll_{k}_acts", W + f"/roll_{k}_frames"
    index = []
    with open(W + f"/roll_{k}.jsonl", "w") as f:
        for r in res:
            v = practice().get((r["suite"], int(r["variant_id"])), {})
            o = {"suite": r["suite"], "id": r["variant_id"], "seed": r["seed"], "model": "edited" if r["arm"] == "edit" else "original",
                 "category": v.get("category"), "difficulty": v.get("difficulty"), "name": v.get("name"),
                 "success": bool(r.get("success")), "steps": r.get("steps"), "termination": r.get("termination"),
                 "instruction": r.get("instruction"), "actions_npy": r.get("actions")}
            if r.get("activations"):
                os.makedirs(acts_dir, exist_ok=True)
                dst = f"{acts_dir}/{r['suite']}_v{r['variant_id']}_s{r['seed']}_{o['model']}.npz"
                shutil.copy(r["activations"], dst); o["activations"] = dst; index.append({k2: o[k2] for k2 in ("suite", "id", "seed", "model", "category", "success", "activations")})
            if r.get("frames_dir"):
                os.makedirs(frames_dir, exist_ok=True)
                link = f"{frames_dir}/{r['suite']}_v{r['variant_id']}_s{r['seed']}_{o['model']}"
                if not os.path.exists(link):
                    os.symlink(r["frames_dir"], link)
                o["frames_dir"] = link
            f.write(json.dumps(o) + "\n")
    if index:
        json.dump(index, open(acts_dir + "/index.json", "w"))
    by_arm = collections.defaultdict(list)
    for r in res:
        v = practice().get((r["suite"], int(r["variant_id"])), {})
        r["category"], r["difficulty"] = v.get("category"), v.get("difficulty")
        by_arm["edited" if r["arm"] == "edit" else "original"].append(r)
    S = {"pass": k, "episodes": len(eps), "model": model, "records": f"WORK/roll_{k}.jsonl"}
    if LANG:
        S["language"] = lang
    for arm, rs in by_arm.items():
        S[arm] = {"success": _rate(rs), "by_category": _table(rs, "category"), "by_suite": _table(rs, "suite"),
                  "by_difficulty": _table(rs, "difficulty"),
                  "termination": dict(collections.Counter(r.get("termination") for r in rs)),
                  "mean_steps_success": round(sum(r["steps"] for r in rs if r["success"]) / max(1, sum(r["success"] for r in rs)), 1)}
    if model == "both":
        o = {(r["suite"], r["variant_id"], r["seed"]): r["success"] for r in by_arm["original"]}
        d = [int(r["success"]) - int(o[(r["suite"], r["variant_id"], r["seed"])]) for r in by_arm["edited"]
             if (r["suite"], r["variant_id"], r["seed"]) in o]
        n = len(d); m = sum(d) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, n - 1))
        S["paired"] = {"n": n, "diff_pp": round(100 * m, 1), "se_pp": round(100 * sd / math.sqrt(n), 1),
                       "wrong_to_right": d.count(1), "right_to_wrong": d.count(-1)}
        byc = collections.defaultdict(list)
        for r in by_arm["edited"]:
            kk = (r["suite"], r["variant_id"], r["seed"])
            if kk in o:
                byc[r["category"]].append(int(r["success"]) - int(o[kk]))
        S["paired_by_category"] = {c: f"{100 * sum(x) / len(x):+.1f} pp (n={len(x)})" for c, x in sorted(byc.items())}
    if index:
        S["activations"] = f"WORK/roll_{k}_acts/ (index.json; keys mean::action.<i>.<what>::action [1024], mean::video.<i>.<what>::agent|wrist [3072])"
    if req.get("save_frames"):
        S["frames"] = (f"WORK/roll_{k}_frames/<suite>_v<id>_s<seed>_<model>/tNNNN.png (agent|wrist side by side) and "
                       "tNNNN_inputs.pt (replayable policy inputs for probe)")
    return json.dumps(S, indent=1)


def probe(req, R, k):
    W = R + "/work"
    model = req.get("model", "original")
    arms = {"original": ["orig"], "edited": ["edit"], "both": ["orig", "edit"]}.get(model)
    if arms is None:
        raise ValueError("model must be original, edited or both")
    inp = req.get("inputs")
    inp = [inp] if isinstance(inp, str) else list(inp or [])
    paths = []
    for p in inp:
        p = p.replace("WORK", W, 1) if p.startswith("WORK") else p
        rp = os.path.realpath(p)
        if not (rp.startswith(os.path.realpath(W) + "/") and rp.endswith("_inputs.pt")):
            raise ValueError(f"input not from your rollouts: {p}")
        if not os.path.exists(p):
            raise ValueError(f"no such input: {p}")
        paths.append(p)
    if not paths or len(paths) > MAX_PROBE:
        raise ValueError(f"1..{MAX_PROBE} inputs required")
    ret = req.get("return", {})
    att = ret.get("attn_layers") or (ret.get("layers") if ret.get("what") == "attn_maps" else None)
    if att and (len(paths) > 4 or len(att) > 4):
        raise ValueError("attention maps: at most 4 inputs and 4 layers")
    rec = None
    if ret.get("layers") and ret.get("what") != "attn_maps":
        what = ret.get("what", "attn_out"); what = what if isinstance(what, list) else [what]
        rec = {"action_blocks": list(ret.get("layers", [])), "video_blocks": list(ret.get("video_layers", [])), "what": what, "per_call": True}
    override = None
    if "edit" in arms:
        ed = W + "/edited"
        if not (os.path.isdir(ed) and os.listdir(ed)):
            raise ValueError("no edits saved in WORK/edited")
        override = _snapshot(ed, R, f"probe_{k}")
    out = W + f"/probe_{k}"
    res = C.probe(paths, out, sock=random.choice(ready_socks()), weights_override=override, arms=arms, record=rec, attention_layers=att)
    if not res.get("ok"):
        raise RuntimeError(f"probe failed: {str(res)[:1000]}")
    json.dump(res.get("results", []), open(out + "/results.json", "w"))
    return json.dumps({"pass": k, "inputs": len(paths), "model": model, "out": f"WORK/probe_{k}/results.json",
                       "fields": "per input+arm: action_norm .npy [32,7] (model space), action .npy [32,7] (executed), "
                                 "activations .npz (if layers), attention {layer: .npy [24 heads, 32 action tokens, 98 video tokens]} "
                                 "(video token t = row t//14, col t%14 of the 7x14 latent grid; cols 0-6 agent view, 7-13 wrist)"}, indent=1)


def forward(req, R, k, target=None):
    t = req.get("type")
    if t == "rollout":
        return rollout(req, R, k)
    if t == "probe":
        return probe(req, R, k)
    raise ValueError("type must be rollout or probe")
