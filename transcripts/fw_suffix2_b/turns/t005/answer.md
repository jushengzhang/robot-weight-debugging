I’ll use a small, balanced raw/clean diagnostic set to locate divergence, while keeping larger validation sets untouched. The splits are fixed now so that later success-rate checks are not selected based on which variants favor an edit.```python
def hkey(v, salt):
    return hashlib.sha256(f"{salt}|{v['suite']}|{v['id']}".encode()).hexdigest()
used = set()
splits = {}
def take_balanced(label, n, language=False, seedbase=2000):
    eligible = [v for v in pv if (v["category"] == "Language Instructions") == language
                and (v["suite"], v["id"]) not in used]
    buckets = defaultdict(list)
    for v in eligible:
        buckets[(v["category"], v["suite"])].append(v)
    for k in buckets:
        buckets[k].sort(key=lambda v: hkey(v, label))
    keys = sorted(buckets)
    chosen = []
    while len(chosen) < n:
        for k in keys:
            if buckets[k] and len(chosen) < n:
                chosen.append(buckets[k].pop())
    for v in chosen:
        used.add((v["suite"], v["id"]))
    splits[label] = [dict(v, seed=seedbase + i) for i, v in enumerate(chosen)]
    return splits[label]
take_balanced("diagnostic", 24, seedbase=2100)
take_balanced("language_diag", 24, language=True, seedbase=2200)
take_balanced("development", 96, seedbase=2300)
for i in range(4):
    take_balanced(f"validation_{i}", 128, seedbase=3000 + 200*i)
    take_balanced(f"language_{i}", 128, language=True, seedbase=4000 + 200*i)
(W / "splits.json").write_text(json.dumps(splits, indent=2))
def eps(label):
    return [{"suite": v["suite"], "id": v["id"], "seed": v["seed"]} for v in splits[label]]
print("Diagnostic episodes:", json.dumps(eps("diagnostic")))
print("Category counts:", dict(Counter(v["category"] for v in pv)))
print("Split sizes:", {k: len(v) for k, v in splits.items()})
with (W / "notes.md").open("a") as f:
    f.write("\nFixed splits in splits.json: diagnostic24, language_diag24, development96, "
            "4x128 tagged validation and 4x128 language validation, all disjoint.\n")
```
