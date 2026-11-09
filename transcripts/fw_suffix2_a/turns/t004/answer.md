I will establish a new matched diagnostic set and reserve separate development and validation sets now. The initial observations from the raw and stripped rollouts will provide a controlled comparison: identical images and robot state, with only the instruction changed.
```python
import hashlib
torch.set_num_threads(min(16, torch.get_num_threads()))
practice = practice_raw
cats = sorted({p["category"] for p in practice})
suites = sorted({p["suite"] for p in practice})
print("COUNTS:", dict(collections.Counter(p["category"] for p in practice)))
used = set()
def pick_balanced(label, n, categories):
    buckets = collections.defaultdict(list)
    for p in practice:
        key = (p["suite"], p["id"])
        if key not in used and p["category"] in categories:
            h = hashlib.sha256((label + ":" + str(key)).encode()).hexdigest()
            buckets[(p["category"], p["suite"])].append((h,p))
    for v in buckets.values():
        v.sort(key=lambda a:a[0])
    out = []
    keys = sorted(buckets)
    while len(out) < n:
        changed = False
        for k in keys:
            if buckets[k] and len(out)<n:
                _,p = buckets[k].pop(0)
                used.add((p["suite"],p["id"]))
                q = dict(p)
                q["seed"] = int(hashlib.sha256((label+str(p["id"])+p["suite"]).encode()).hexdigest()[:7],16)%100000
                out.append(q)
                changed = True
        if not changed:
            raise RuntimeError("insufficient variants")
    return out
tagcats = [c for c in cats if c != "Language Instructions"]
splits = {}
splits["diagnostic"] = pick_balanced("new-diag",48,tagcats)
splits["language_diag"] = pick_balanced("new-ldiag",16,["Language Instructions"])
splits["development"] = pick_balanced("new-dev",96,tagcats)
splits["language_dev"] = pick_balanced("new-ldev",48,["Language Instructions"])
for i in range(6):
    splits[f"tag_validation_{i}"] = pick_balanced(f"new-tval-{i}",128,tagcats)
for i in range(4):
    splits[f"language_validation_{i}"] = pick_balanced(f"new-lval-{i}",128,["Language Instructions"])
(W/"splits.json").write_text(json.dumps(splits,indent=2))
def episodes(key):
    return [{k:p[k] for k in ("suite","id","seed")} for p in splits[key]]
diag_episodes = episodes("diagnostic")+episodes("language_diag")
(W/"diagnostic_episodes.json").write_text(json.dumps(diag_episodes))
notes = """# New session: FastWAM suffix invariance
## Constraints and prior findings
50 forward passes; no training/local network execution. Only practice metadata read.
Earlier mean-contrast gates and projections failed to restore clean actions. Prior candidate is NOT applied.
Prior notes report +~1.8pp tagged with weak significance and >2pp paraphrase loss; not a baseline to retain automatically.
## Splits
splits.json: fixed balanced diagnostic48 tagged+16 language; development96 tagged+48 language;
six disjoint tagged validation sets of128, four language validation sets of128. Selection independent of outcomes.
## Plan
Matched raw/strip diagnostics, improve token-level attribution rather than relying only on context means;
causal expert/pathway and dose comparisons; freeze and validate paired success and safety.
## Passes
None yet.
## Current edit
Original (no edits).
"""
(W/"notes.md").write_text(notes)
print("DIAGNOSTIC EPISODES:", json.dumps(diag_episodes))
print("\nPRIOR CANDIDATE FILE COUNT:", len(list((W/"prior_candidate").glob("*.npy"))))
```
