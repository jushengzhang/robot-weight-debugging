The shared-attention projections use 24 contiguous 128-dimensional heads in both experts, which allows head-specific edits. Next I will inspect the variant naming scheme and create a fixed practice-data split so that later validation does not reuse the variants used for diagnosis.

```python
variants = raw_variants
bycat = defaultdict(list)
bykey = {}
for v in variants:
    bycat[v["category"]].append(v)
    bykey[(v["suite"], int(v["id"]))] = v
print("Counts by category:", {k: len(v) for k, v in bycat.items()})
print("Counts by suite:", dict(Counter(v["suite"] for v in variants)))
for cat, vs in sorted(bycat.items()):
    print("\n", cat, "examples:")
    for v in vs[:3]:
        print(v)
    print("difficulty:", dict(sorted(Counter(v["difficulty"] for v in vs).items())))

def split_of(v):
    h = hashlib.sha256(f"practice-split-v1:{v['suite']}:{v['id']}".encode()).digest()
    return "diagnosis" if int.from_bytes(h[:4], "little") % 3 == 0 else "validation"
for v in variants:
    v["_split"] = split_of(v)
print("\nSplit:", dict(Counter((v["category"], v["_split"]) for v in variants)))
(W / "practice_split.json").write_text(json.dumps(
    [{"suite": v["suite"], "id": v["id"], "split": v["_split"]} for v in variants]))
```
