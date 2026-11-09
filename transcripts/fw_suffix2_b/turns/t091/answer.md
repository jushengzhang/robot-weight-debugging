Only 140 unused paraphrase variants remain after the reserved and development sets. To keep validation on distinct variants rather than repeat seeds, I’ll use 1,024 tagged episodes, 768 matched clean episodes, and 640 paraphrase episodes. I’ll select the additional variants by a fixed hash now and verify that the frozen weights exactly match the saved candidate.
```python
def file_sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for z in iter(lambda: f.read(8*1024*1024), b""):
            h.update(z)
    return h.hexdigest()

expected = {p.name[:-4] for p in base.glob("*.npy")}
assert set(edited()) == expected
manifest = {}
for n in sorted(expected):
    h = file_sha(W/"edited"/(n+".npy"))
    assert h == file_sha(base/(n+".npy")), n
    manifest[n] = h
(W/"frozen_manifest.json").write_text(json.dumps(manifest,indent=2))

used = {(r["suite"],int(r["id"])) for split in sp.values() for r in split}
used |= {(r["suite"],int(r["id"])) for r in langdev}
suites = sorted({v["suite"] for v in variants})
vs = dict(sp)

def choose_balanced(n, categories, salt, seedbase, rotation=0):
    groups = {(c,s): [] for c in sorted(categories) for s in suites}
    for v in variants:
        key = (v["suite"],int(v["id"]))
        if key not in used and v["category"] in categories:
            groups[(v["category"],v["suite"])].append(v)
    for g in groups:
        groups[g].sort(key=lambda v: hashlib.sha256(
            f'{salt}|{v["suite"]}|{v["id"]}'.encode()).digest())
    order = list(groups)
    out = []
    j = rotation
    while len(out) < n:
        g = order[j % len(order)]
        j += 1
        if not groups[g]:
            if not any(groups.values()):
                raise ValueError("Not enough unused practice variants")
            continue
        v = dict(groups[g].pop(0))
        v["seed"] = seedbase + len(out)
        out.append(v)
        used.add((v["suite"],int(v["id"])))
    return out

rawcats = sorted({v["category"] for v in variants} - {"Language Instructions"})
for k in range(4,8):
    vs[f"validation_{k}"] = choose_balanced(
        128, rawcats, f"frozen-r256-raw-{k}", 60000+1000*k, rotation=(8*k)%24)
vs["language_4"] = choose_balanced(
    128, ["Language Instructions"], "frozen-r256-language-4", 74000)

plan = []
def add_plan(split, language):
    rs = [{k:r[k] for k in ("suite","id","seed")} for r in vs[split]]
    plan.append({"pass":32+len(plan), "split":split, "condition":language,
                 "request":{"type":"rollout","model":"both","language":language,"episodes":rs}})
for i in range(5):
    add_plan(f"validation_{i}","raw")
    add_plan(f"validation_{i}","strip")
    add_plan(f"language_{i}","raw")
add_plan("validation_5","raw")
add_plan("validation_5","strip")
add_plan("validation_6","raw")
add_plan("validation_7","raw")
assert len(plan)==19 and plan[-1]["pass"]==50
all_validation = [r for k,v in vs.items() if k.startswith(("validation_","language_")) and k!="language_diag" for r in v]
assert len({(r["suite"],r["id"]) for r in all_validation}) == len(all_validation)
(W/"validation_splits.json").write_text(json.dumps(vs,indent=2))
(W/"validation_plan.json").write_text(json.dumps(plan,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\nValidation plan finalized before any validation outcomes: "
            "8x128 tagged (1024),6x128 matched strip (768),5x128 language (640). "
            "Only140 unused language variants available; chose128 by fixed hash rather "
            "than repeat seeds. All validation variants disjoint from diagnostics and "
            "development. Saved validation_splits.json, validation_plan.json. "
            "All308 edited files SHA256-identical to frozen baseline candidate; hashes "
            "in frozen_manifest.json. No further candidate changes.\n")
print("Frozen tensors verified:",len(manifest))
print("Validation totals: raw=1024, strip=768, language=640; all distinct within condition.")
print(json.dumps(plan[0]["request"],separators=(",",":")))
```
