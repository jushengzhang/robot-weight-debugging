I will identify how task names and perturbation levels are encoded in the practice list. This will let me compare paraphrases with easier instructions for the same tasks, especially in the shared-scene goal suite where visual context cannot identify the task.
```python
variants = variants_raw
bykey = {(v["suite"], v["id"]): v for v in variants}
print("counts by category/suite")
for cat in sorted({v["category"] for v in variants}):
    print(cat, dict(collections.Counter(v["suite"] for v in variants if v["category"] == cat)))
langvars = [v for v in variants if v["category"] == "Language Instructions"]
print("language difficulty", collections.Counter((v["suite"], v["difficulty"]) for v in langvars))
for suite in sorted({v["suite"] for v in variants}):
    vs = [v for v in langvars if v["suite"] == suite]
    print("\n", suite, "first", vs[:5], "last", vs[-3:])
print("initial edit count", len(edited()))
```
