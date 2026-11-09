I will use a small, balanced diagnostic set covering all six tag-bearing categories, with the same variants and seeds for raw and stripped instructions. I will also reserve disjoint practice sets for later paired validation and language-preservation checks.
```python
import hashlib
cats = sorted({v["category"] for v in variants})
suites = sorted({v["suite"] for v in variants})
print("Categories:", cats)
print("Suites:", suites)
print("Counts:", dict(collections.Counter(v["category"] for v in variants)))

def stable_order(v, salt):
    return hashlib.sha256(f'{salt}:{v["suite"]}:{v["id"]}'.encode()).hexdigest()

tag_cats = [c for c in cats if c != "Language Instructions"]
diag = []
for c in tag_cats:
    for s in suites:
        pool = sorted([v for v in variants if v["category"] == c and v["suite"] == s],
                      key=lambda v: stable_order(v, "diagnostic"))
        if pool:
            diag.append(pool[0])
diag_keys = {(v["suite"], v["id"]) for v in diag}
remaining = [v for v in variants if (v["suite"], v["id"]) not in diag_keys]

def balanced_set(pool, n, salt):
    buckets = collections.defaultdict(list)
    for v in pool:
        buckets[(v["category"], v["suite"])].append(v)
    for k in buckets:
        buckets[k].sort(key=lambda v: stable_order(v, salt))
    keys = sorted(buckets, key=lambda k: hashlib.sha256((salt+str(k)).encode()).hexdigest())
    result = []
    while len(result) < n:
        for k in keys:
            if buckets[k] and len(result) < n:
                result.append(buckets[k].pop(0))
    return result

splits = {"diagnostic": diag}
used = set(diag_keys)
for name, n, language, salt in [
    ("development", 96, False, "dev"),
    ("validation_a", 128, False, "val-a"),
    ("validation_b", 128, False, "val-b"),
    ("language_a", 128, True, "lang-a"),
    ("language_b", 128, True, "lang-b"),
]:
    pool = [v for v in variants if (v["suite"], v["id"]) not in used
            and (v["category"] == "Language Instructions") == language]
    splits[name] = balanced_set(pool, n, salt)
    used.update((v["suite"], v["id"]) for v in splits[name])
(work / "splits.json").write_text(json.dumps(splits, indent=2))

diag_episodes = [{"suite": v["suite"], "id": v["id"], "seed": 1701+i}
                 for i, v in enumerate(diag)]
print("Diagnostic episodes:", json.dumps(diag_episodes))
print("Diagnostic variants:")
for v in diag:
    print(v["suite"], v["id"], v["category"], v["difficulty"], v["name"])
with (work / "notes.md").open("a") as f:
    f.write("\n## Fixed splits\nSaved in splits.json: diagnostic=24, development=96, "
            "validation_a/b=128 each (tagged categories), language_a/b=128 each. "
            "All disjoint, selected deterministically by hashes and category/suite balancing.\n")
```
