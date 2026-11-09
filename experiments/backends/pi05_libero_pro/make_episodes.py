#!/usr/bin/env python
"""Enumerate all LIBERO-PRO evaluation episodes (paper's 5 perturbation types) + the practice/held-out split.
  python make_episodes.py <LIBERO-PRO root> <out dir>
Episode = (suite, perturbation type, task, init_idx 0..49); seed 7. Key '<suite>/<ptype>/<task>/<init:02d>'.
split: int(sha256(key), 16) % 2 == 0 -> practice, else held-out (written chmod 600).
instruction = what the servers feed by default (language="pro"). Also writes orig_episodes.jsonl (unperturbed LIBERO suites from the same copy; reference only, not split)."""
import difflib, hashlib, json, os, re, sys
from pathlib import Path

root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(root))
from libero.libero.benchmark.libero_suite_task_map import libero_task_map  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
PTYPES = {"object": "object", "position": "swap", "semantic": "lan", "task": "task", "environment": "env"}
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}  # openpi/LIBERO-PRO README
B = root / "libero/libero/bddl_files"
I = root / "libero/libero/init_files"
env_man = json.loads((B / "gen_env_manifest.json").read_text())


def lang(text):
    m = re.search(r"\(:language\s+(.+?)\)", text, flags=re.S)
    return " ".join(m.group(1).split()) if m else None


def block(text, head):
    s = text.find(head)
    if s < 0:
        return ""
    d = 0
    for i in range(s, len(text)):
        d += (text[i] == "(") - (text[i] == ")")
        if d == 0:
            return " ".join(text[s:i + 1].split())
    return ""


def fname_lang(task):  # = upstream grab_language_from_filename (what run_libero_eval feeds as task.language)
    x = task + ".bddl"
    if x[0].isupper():
        k = 8 if "SCENE10" in x else 7
        s = " ".join(x[x.find("SCENE") + k:].split("_"))
    else:
        s = " ".join(x.split("_"))
    return s[: s.find(".bddl")]


def detail(orig, pert, ptype):
    if ptype in ("semantic", "task"):
        d = {"language": lang(pert)}
        if ptype == "task":
            d["goal"] = block(pert, "(:goal")
            d["orig_goal"] = block(orig, "(:goal")
        return d
    if ptype == "position":
        ia, ib = block(orig, "(:init"), block(pert, "(:init")
        a = dict(re.findall(r"\(On (\S+) (\S+)\)", ia)); b = dict(re.findall(r"\(On (\S+) (\S+)\)", ib))
        return {"moved": {k: [a.get(k), v] for k, v in b.items() if a.get(k) != v}}
    if ptype == "object":
        oa = set(re.findall(r"\b(\w+_\d+)\b", block(orig, "(:objects")))
        ob = set(re.findall(r"\b(\w+_\d+)\b", block(pert, "(:objects")))
        return {"removed": sorted(oa - ob), "added": sorted(ob - oa)}
    if ptype == "environment":
        m = re.search(r"\(problem (\S+)\)", pert)
        return {"problem": m.group(1) if m else None}
    return {}


eps, orig_eps, check = [], [], []
for suite in SUITES:
    for t_idx, task in enumerate(libero_task_map[suite]):
        orig = (B / suite / f"{task}.bddl").read_text()
        ol = lang(orig)
        check.append((suite, task, ol.lower().rstrip("."), fname_lang(task)))
        for i in range(50):
            orig_eps.append({"key": f"{suite}/original/{task}/{i:02d}", "suite": suite, "ptype": "original",
                             "pro_suite": suite, "task": task, "task_idx": t_idx, "init_idx": i, "seed": 7,
                             "max_steps": MAX_STEPS[suite], "instruction": fname_lang(task),
                             "instruction_orig": fname_lang(task)})
        for ptype, suf in PTYPES.items():
            ps = f"{suite}_{suf}"
            names = libero_task_map[ps]
            assert task in names, (ps, task)
            pert = (B / ps / f"{task}.bddl").read_text()
            assert (I / ps / f"{task}.pruned_init").exists(), (ps, task)
            pl = lang(pert)
            changed = pl.lower().rstrip(".") != ol.lower().rstrip(".")
            # fed instruction ('pro' mode): the perturbed BDDL language where LIBERO-PRO changed it (semantic/task),
            # else the standard LIBERO instruction (task.language, what pi05_libero/FastWAM were trained on)
            instr = pl.lower().rstrip(".") if changed else fname_lang(task)
            det = detail(orig, pert, ptype)
            extra = {}
            if ptype == "environment":
                extra["env_identical_to_orig"] = env_man[f"{suite}/{task}"]["identical_to_orig"]
            for i in range(50):
                key = f"{suite}/{ptype}/{task}/{i:02d}"
                eps.append({"key": key, "suite": suite, "ptype": ptype, "pro_suite": ps, "task": task,
                            "task_idx": names.index(task), "init_idx": i, "seed": 7, "max_steps": MAX_STEPS[suite],
                            "instruction": instr, "instruction_bddl": pl,
                            "instruction_orig": fname_lang(task), "instruction_changed": changed,
                            "bddl": f"{ps}/{task}.bddl", "bddl_sha256": hashlib.sha256(pert.encode()).hexdigest()[:16],
                            "detail": det, **extra})

out.mkdir(parents=True, exist_ok=True)
bad = [c for c in check if c[2] != c[3]]
print("orig BDDL language (lowercased) != filename language for", len(bad), "tasks:", bad[:5])
w = lambda p, rows: p.write_text("".join(json.dumps(r) + "\n" for r in rows))
w(out / "all_episodes.jsonl", eps)
w(out / "orig_episodes.jsonl", orig_eps)
prac = [e for e in eps if int(hashlib.sha256(e["key"].encode()).hexdigest(), 16) % 2 == 0]
held = [e for e in eps if int(hashlib.sha256(e["key"].encode()).hexdigest(), 16) % 2 == 1]
w(out / "practice.jsonl", prac)
hp = out / "heldout.jsonl"
if hp.exists():
    os.chmod(hp, 0o600)
fd = os.open(hp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    f.write("".join(json.dumps(r) + "\n" for r in held))
os.chmod(hp, 0o600)
import collections
c = collections.Counter((e["ptype"], e["suite"]) for e in eps)
cp = collections.Counter(e["ptype"] for e in prac)
print("total", len(eps), "practice", len(prac), "heldout", len(held), "orig", len(orig_eps))
for p in PTYPES:
    print(p, sum(v for (pp, s), v in c.items() if pp == p), "practice", cp[p],
          {s: c[(p, s)] for s in SUITES})
print("instruction_changed:", collections.Counter((e["ptype"], e["instruction_changed"]) for e in eps if e["init_idx"] == 0))
