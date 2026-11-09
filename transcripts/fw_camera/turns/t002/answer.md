I will separate discovery and validation variants before looking at outcomes. I will also inspect the camera-variant naming scheme and the experts' attention tensors so the initial rollouts can compare the same tasks under camera changes and mild control perturbations.```python
import hashlib
practice = practice_raw
cats = sorted(set(x["category"] for x in practice))
print("Counts by category:", dict(collections.Counter(x["category"] for x in practice)))
print("Counts by suite:", dict(collections.Counter(x["suite"] for x in practice)))
for cat in cats:
    sub = [x for x in practice if x["category"] == cat]
    print("\n", cat)
    for x in sub[:3]:
        print(x)
print("\nNon-block tensors:")
for n in NAMES:
    if ".blocks." not in n:
        print(n, tuple(SHAPES[n]))
print("\nVideo block 0:")
for n in NAMES:
    if ".video.blocks.0." in n:
        print(n, tuple(SHAPES[n]))
def variant_partition(x):
    s = f'camera-repair-v1/{x["suite"]}/{x["id"]}'
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16) % 5
discovery = [x for x in practice if variant_partition(x) < 2]
validation = [x for x in practice if variant_partition(x) >= 2]
(W / "partition.json").write_text(json.dumps({
    "discovery": [{"suite": x["suite"], "id": x["id"]} for x in discovery],
    "validation": [{"suite": x["suite"], "id": x["id"]} for x in validation],
}, indent=2))
print("\nDiscovery/validation cameras:",
      sum(x["category"] == "Camera Viewpoints" for x in discovery),
      sum(x["category"] == "Camera Viewpoints" for x in validation))
```
