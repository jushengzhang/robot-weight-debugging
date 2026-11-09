The current edit changes only the token-embedding table. I will recover its exact lexical substitutions and compare the half- and full-strength destination edits, checking whether their gains come from the intended paraphrases rather than unrelated rollout differences.
```python
from collections import defaultdict, Counter
import math
torch.set_num_threads(min(16, os.cpu_count() or 16))
E_NAME = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
specs = {}
for p in sorted(W.glob("*spec.json")):
    specs[p.stem] = json.loads(p.read_text())
    print(p.name, json.dumps(specs[p.stem], ensure_ascii=True))

def read_roll(k):
    return [json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]

def pair_roll(k):
    pairs = defaultdict(dict)
    for r in read_roll(k):
        pairs[(r["suite"], r["id"], r["seed"])][r["model"]] = r
    return [v for v in pairs.values() if "original" in v and "edited" in v]

def paired_stats(ps):
    ds = np.array([int(bool(p["edited"]["success"]))-int(bool(p["original"]["success"])) for p in ps])
    return dict(n=len(ps),
                original=sum(bool(p["original"]["success"]) for p in ps),
                edited=sum(bool(p["edited"]["success"]) for p in ps),
                fixes=int((ds>0).sum()), regressions=int((ds<0).sum()),
                diff_pp=round(100*ds.mean(), 2) if len(ds) else None,
                se_pp=round(100*ds.std(ddof=1)/np.sqrt(len(ds)), 2) if len(ds)>1 else None,
                errors=Counter((p["original"]["termination"], p["edited"]["termination"])
                               for p in ps if "error" in (p["original"]["termination"], p["edited"]["termination"])))

for k in (13, 14, 15, 16, 17):
    ps = pair_roll(k)
    print("\nPASS", k, paired_stats(ps))
    if k in (15, 16, 17):
        for p in ps:
            a,b = p["original"],p["edited"]
            if bool(a["success"]) != bool(b["success"]):
                print("FIX" if b["success"] else "REG", a["suite"], a["id"],
                      f"d{a['difficulty']}", repr(a["instruction"] or b["instruction"]),
                      (a["termination"], b["termination"]))

print("\nEdited artifact:")
for p in (W/"edited").iterdir():
    a=np.load(p, mmap_mode="r")
    print(p.name, a.shape, a.dtype, p.stat().st_size)
print("\nProbe artifact examples:")
for k in (3,5,6,8,11,12):
    d=W/f"probe_{k}"
    print(k, [p.name for p in list(d.iterdir())[:5]])
    ip=d/"index.json"
    if ip.exists():
        idx=json.loads(ip.read_text())
        print(str(idx)[:650])
```
