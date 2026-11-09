#!/usr/bin/env python
"""Repair LIBERO-Recovery scene BDDLs (released data bugs; originals untouched, fixed copies -> data/scenes_fixed/).
  python fix_scenes.py samples/in.jsonl samples/out.jsonl
Fixes (each recorded in the episode as 'bddl_fixes'):
  floor_target : libero_floor_manipulation (libero_object) scenes declare every init region with (:target main_table),
                 but the floor arena has no main_table -> KeyError('main_table'). -> (:target floor)
  goal_regions : the scene writer only keeps init regions; table regions used only by the :goal (e.g.
                 stove_front_region) are dropped -> KeyError('main_table_stove_front_region') at check_success.
                 -> copy the missing region blocks from the original LIBERO task BDDL.
The init-region placement is discarded anyway (set_init_state), so neither fix changes the recorded state."""
import hashlib, json, re, sys
from pathlib import Path
R = Path(__file__).resolve().parent
ORIG = Path("/mnt/SSD_16TB/jusheng/wsp_pi05/third_party/LIBERO-orig/libero/libero/bddl_files")
OUT = R / "data/scenes_fixed"


def block_span(text, start):
    """index of the '(' at start -> end index (exclusive) of the balanced block"""
    d = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            d += 1
        elif text[i] == ")":
            d -= 1
            if d == 0:
                return i + 1
    raise ValueError("unbalanced")


def regions_block(text):
    i = text.index("(:regions")
    return i, block_span(text, i)


def region_names(text):
    i, j = regions_block(text)
    body = text[i + len("(:regions"): j - 1]
    names, k = {}, 0
    while True:
        m = re.compile(r"\(\s*([A-Za-z0-9_]+)\s*\n").search(body, k)
        if not m:
            break
        e = block_span(body, m.start())
        names[m.group(1)] = body[m.start(): e]
        k = e
    return names


def fix(text, suite, task_dir):
    fixes = []
    if "libero_floor_manipulation" in text and "(:target main_table)" in text:
        text = text.replace("(:target main_table)", "(:target floor)")
        fixes.append("floor_target")
    have = region_names(text)
    gi = text.find("(:goal")
    goal = text[gi: block_span(text, gi)] if gi >= 0 else ""
    table = "floor" if "libero_floor_manipulation" in text else "main_table"
    need = sorted({t[len(table) + 1:] for t in re.findall(rf"\b{table}_[A-Za-z0-9_]+", goal)} - set(have))
    if need:
        cands = list(ORIG.glob(f"*/{task_dir}.bddl"))
        src = (ORIG / suite / f"{task_dir}.bddl") if (ORIG / suite / f"{task_dir}.bddl").exists() else (cands[0] if cands else None)
        if src is None:
            raise FileNotFoundError(f"no original bddl for {task_dir}")
        orig = region_names(src.read_text())
        missing = [n for n in need if n not in orig]
        if missing:
            raise KeyError(f"regions {missing} not in {src}")
        i, j = regions_block(text)
        ins = "".join("      " + orig[n].strip() + "\n" for n in need)
        text = text[: j - 1].rstrip() + "\n" + ins + "    " + text[j - 1:]
        fixes.append("goal_regions:" + ",".join(need))
    return text, fixes


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    eps = [json.loads(l) for l in open(sys.argv[1])]
    n = 0
    for e in eps:
        b = Path(e.get("bddl_orig", e["bddl"]))
        e["bddl_orig"] = str(b)
        e.setdefault("sim_state", str(b)[:-5] + "_sim_state.npy")
        task_dir = b.parent.parent.name
        text, fixes = fix(b.read_text(), e["suite"], task_dir)
        e["bddl_fixes"] = fixes
        if fixes:
            fb = OUT / (hashlib.sha1(str(b).encode()).hexdigest()[:16] + "_" + b.name)
            fb.write_text(text)
            e["bddl"] = str(fb)
            n += 1
        else:
            e["bddl"] = str(b)
    with open(sys.argv[2], "w") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")
    import collections
    print(n, "of", len(eps), "fixed;", collections.Counter(x.split(":")[0] for e in eps for x in e["bddl_fixes"]))
