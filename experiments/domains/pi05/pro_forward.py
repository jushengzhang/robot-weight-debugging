"""Forward-pass tool for the pi0.5 x LIBERO-PRO domain (recover_forward.py with LIBERO-PRO episodes)."""
import collections, json, math, os, random, shutil, sys

F = "/mnt/SSD_16TB/jusheng/wsp_pro"
BENCH = "/mnt/SSD_16TB/jusheng/wsp_pro/wsp/bench"
ALL = "/mnt/SSD_16TB/jusheng/wsp_pro/episodes/practice.jsonl"
sys.path.insert(0, F)
import client as C  # noqa: E402
SOCKS = F + "/run/sock/*.sock"

MAX_EP, MAX_EP_BOTH, MAX_PROBE = 256, 128, 512
LANG = os.environ.get("WSP_LANGUAGE")  # set (raw) only for the instruction-suffix domain; other runs always use strip
_PRAC = None


def _snapshot(ed, R, tag):
    """Copy WORK/edited to tmpfs: every server re-reads the override for each episode, and the run disk reads only
    a few MB/s under load (a 3 GB edit took minutes per episode). Older snapshots of the same run are removed."""
    root = "/dev/shm/wsp_pi_" + os.path.basename(os.path.abspath(R))
    os.makedirs(root, exist_ok=True)
    for old in os.listdir(root):
        shutil.rmtree(os.path.join(root, old), ignore_errors=True)
    dst = os.path.join(root, tag + "_weights")
    shutil.copytree(ed, dst)
    return dst


def ready_socks():
    return [s for s in C.list_socks(SOCKS) if os.path.exists(s[:-5] + ".ready")]


_EPS = None


def practice():
    global _PRAC, _EPS
    if _PRAC is None:
        _PRAC = {int(v["id"]): v for v in json.load(open(BENCH + "/practice_episodes.json"))}
        _EPS = {}
        for l in open(ALL):
            e = json.loads(l)
            _EPS[e["key"]] = e
    return _PRAC


def _ep(key, seed):
    practice()
    e = dict(_EPS[key])
    e["seed"] = int(seed)
    return e


def _episodes(req):
    P = practice()
    if "episodes" in req:
        eps = []
        for e in req["episodes"]:
            i = int(e.get("id", -1))
            if i not in P:
                raise ValueError(f"id {i} is not a practice episode (only BENCH/practice_episodes.json may be run)")
            eps.append(_ep(P[i]["key"], e.get("seed", 7)))
        return eps
    s = req.get("sample")
    if not isinstance(s, dict) or "n" not in s:
        raise ValueError("give 'episodes' or 'sample' {ptype?, suite?, task?, n, seed}")
    pt = s.get("ptype"); pt = [pt] if isinstance(pt, str) else pt
    pool = [v for v in P.values() if (not pt or v["ptype"] in pt) and (not s.get("suite") or v["suite"] == s["suite"])
            and (not s.get("task") or v["task"] == s["task"])]
    if not pool:
        raise ValueError("no practice episode matches the sample filter")
    rng = random.Random(int(s.get("seed", 0)))
    pick = rng.sample(sorted(pool, key=lambda v: v["id"]), min(int(s["n"]), len(pool)))
    return [_ep(v["key"], s.get("episode_seed", 7)) for v in pick]


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
        rec = {"vlm_layers": list(r.get("vlm_layers", [])), "expert_layers": list(r.get("expert_layers", [])),
               "vision_layers": list(r.get("vision_layers", [])), "what": what}
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
    n_err = len(errs)
    res = [r for r in res if r.get("termination") != "error"]  # scene construction failures are not policy outcomes
    if any(r.get("arm") == "edit" and r.get("restore_verified") is False for r in res):
        raise RuntimeError("weight restore verification failed")
    # editor-facing records
    sid2id = {v["key"]: i for i, v in practice().items()}
    acts_dir, frames_dir = W + f"/roll_{k}_acts", W + f"/roll_{k}_frames"
    index = []
    for r in res:
        r["model"] = "edited" if r["arm"] == "edit" else "original"
        r["id"] = sid2id.get(r.get("key"))
        r["stem"] = f"e{r['id']}_seed{r['seed']}_{r['model']}"
        r["loop"] = (not r.get("success")) and (r.get("grip_flips") or 0) >= 6
    with open(W + f"/roll_{k}.jsonl", "w") as f:
        for r in res:
            o = {kk: r.get(kk) for kk in ("id", "seed", "model", "suite", "ptype", "task", "instruction", "instruction_orig",
                                          "success", "orig_goal_reached", "steps", "termination", "grip_flips",
                                          "frac_grip_closed", "eef_path_m", "eef_net_m", "obj_moved_m", "actions")}
            o["actions_npy"] = o.pop("actions")
            if r.get("activations"):
                os.makedirs(acts_dir, exist_ok=True)
                dst = f"{acts_dir}/{r['stem']}.npz"
                shutil.copy(r["activations"], dst); o["activations"] = dst
                index.append({k2: o[k2] for k2 in ("id", "seed", "model", "ptype", "success", "activations")})
            if r.get("frames_dir"):
                os.makedirs(frames_dir, exist_ok=True)
                link = f"{frames_dir}/{r['stem']}"
                if not os.path.exists(link):
                    os.symlink(r["frames_dir"], link)
                o["frames_dir"] = link
            f.write(json.dumps(o) + "\n")
    if index:
        json.dump(index, open(acts_dir + "/index.json", "w"))
    by_arm = collections.defaultdict(list)
    for r in res:
        by_arm[r["model"]].append(r)
    S = {"pass": k, "episodes": len(eps), "model": model, "records": f"WORK/roll_{k}.jsonl",
         "scene_errors_excluded": n_err}
    for arm, rs in by_arm.items():
        fails = [r for r in rs if not r.get("success")]
        S[arm] = {"success": _rate(rs), "by_ptype": _table(rs, "ptype"), "by_suite": _table(rs, "suite"),
                  "termination": dict(collections.Counter(r.get("termination") for r in rs)),
                  "did_original_task_instead (task perturbation failures)": sum(bool(r.get("orig_goal_reached")) and not r.get("success") for r in rs),
                  "regrasp_loop_failures (>=6 gripper flips)": f"{sum(r['loop'] for r in rs)}/{len(fails)} failures",
                  "mean_steps_success": round(sum(r["steps"] for r in rs if r["success"]) / max(1, sum(r["success"] for r in rs)), 1)}
    if model == "both":
        key = lambda r: (r.get("key"), r["seed"])
        o = {key(r): r["success"] for r in by_arm["original"]}
        d = [int(r["success"]) - int(o[key(r)]) for r in by_arm["edited"] if key(r) in o]
        n = len(d); m = sum(d) / max(1, n)
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, n - 1))
        S["paired"] = {"n": n, "diff_pp": round(100 * m, 1), "se_pp": round(100 * sd / math.sqrt(max(1, n)), 1),
                       "wrong_to_right": d.count(1), "right_to_wrong": d.count(-1)}
        byl = collections.defaultdict(list)
        for r in by_arm["edited"]:
            if key(r) in o:
                byl[r.get("ptype")].append(int(r["success"]) - int(o[key(r)]))
        S["paired_by_ptype"] = {c: f"{100 * sum(x) / len(x):+.1f} pp (n={len(x)})" for c, x in sorted(byl.items())}
    if index:
        S["activations"] = (f"WORK/roll_{k}_acts/ (index.json; keys mean::vlm.<i>.<what>::base|wrist|lang [2048], "
                            "mean::expert.<i>.<what>::action [1024], mean::vision.<i>.<what>::base|wrist [1152])")
    if req.get("save_frames"):
        S["frames"] = (f"WORK/roll_{k}_frames/e<id>_seed<seed>_<model>/tNNNN.png (agent|wrist side by side) and "
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
    if (ret.get("layers") or ret.get("vlm_layers") or ret.get("vision_layers")) and ret.get("what") != "attn_maps":
        what = ret.get("what", "attn_out"); what = what if isinstance(what, list) else [what]
        rec = {"expert_layers": list(ret.get("layers", [])), "vlm_layers": list(ret.get("vlm_layers", [])),
               "vision_layers": list(ret.get("vision_layers", [])), "what": what, "per_call": True}
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
                       "fields": "per input+arm: action .npy [10,7] (executed space), matches_saved_chunk, activations .npz (if layers), "
                                 "attention {expert layer: .npy [8 heads, 10 action tokens, 968 prefix tokens]} (prefix_layout gives "
                                 "the token ranges: 256 base-image, 256 wrist-image, 256 masked empty image, 200 prompt tokens)"}, indent=1)


def forward(req, R, k, target=None):
    t = req.get("type")
    if t == "rollout":
        return rollout(req, R, k)
    if t == "probe":
        return probe(req, R, k)
    raise ValueError("type must be rollout or probe")
