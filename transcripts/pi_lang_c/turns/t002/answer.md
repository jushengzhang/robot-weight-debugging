I will inspect how the practice list groups language variants by task and difficulty. This will let me compare paraphrases against literal instructions in the same scene, especially in `libero_goal`, where visual context alone cannot identify the requested task.
```python
cats = collections.Counter(v["category"] for v in variants)
print("Categories:", dict(cats))
lang = [v for v in variants if v["category"] == "Language Instructions"]
print("Language counts:", sorted(collections.Counter((v["suite"], v["difficulty"]) for v in lang).items()))
for suite in sorted({v["suite"] for v in variants}):
    lv = [v for v in lang if v["suite"] == suite]
    print("\n", suite, "examples:")
    for v in lv[:6] + [x for x in lv if x["difficulty"] >= 4][:8]:
        print(v)
print("\nGoal non-language examples:")
for v in [v for v in variants if v["suite"] == "libero_goal" and v["category"] != "Language Instructions"][:12]:
    print(v)
print("WORK files:", [p.name for p in W.iterdir()])
```
