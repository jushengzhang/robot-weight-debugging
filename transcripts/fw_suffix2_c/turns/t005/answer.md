I will use 24 diagnostic variants, one from each non-language category and suite. I will also create disjoint development and validation sets now, so that any later success-rate claim is tested on variants that were not used to choose the edit.
```python
used_variants = set()
def hid(s):
    return hashlib.sha256(s.encode()).hexdigest()
def balanced_pick(label, n, language=False):
    pool = [v for v in variants if (v["category"] == "Language Instructions") == language
            and (v["suite"], v["id"]) not in used_variants]
    groups = defaultdict(list)
    for v in pool:
        groups[(v["category"], v["suite"])].append(v)
    for key in groups:
        groups[key].sort(key=lambda v: hid(f"second-session/{label}/{v['suite']}/{v['id']}"))
    keys = sorted(groups)
    result = []
    k = 0
    while len(result) < n:
        g = groups[keys[k % len(keys)]]
        if g:
            v = g.pop()
            used_variants.add((v["suite"], v["id"]))
            vv = dict(v)
            vv["seed"] = int(hid(f"seed/{label}/{v['suite']}/{v['id']}")[:8], 16) % 1000000
            result.append(vv)
        k += 1
    return result
sets = {}
for label, n, lang in [("diagnostic",24,False),("language_diag",32,True),
                       ("development",96,False),
                       ("validation_a",128,False),("validation_b",128,False),
                       ("validation_c",128,False),("validation_d",128,False),
                       ("language_a",128,True),("language_b",128,True),
                       ("language_c",128,True)]:
    sets[label] = balanced_pick(label,n,lang)
(W / "splits.json").write_text(json.dumps(sets, indent=2))
def eps(label):
    return [{k:v[k] for k in ("suite","id","seed")} for v in sets[label]]
print(json.dumps(eps("diagnostic")))
with (W / "notes.md").open("a") as f:
    f.write("\nFixed hash/category/suite-balanced disjoint splits in splits.json: diag24, language_diag32, development96, "
            "tagged validation A-D128 each, language A-C128 each. Diagnostic inputs will be raw/strip matched.\n")
```
