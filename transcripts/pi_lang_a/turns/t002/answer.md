I will group the language variants by task and difficulty. This will let me compare hard paraphrases with easier wording for the same goal, rather than confounding language sensitivity with different scenes or objects.

```python
print("categories:", Counter(v["category"] for v in variants))
print("language counts:", Counter((v["suite"], v["difficulty"]) for v in variants if v["category"] == "Language Instructions"))
goal_lang = [v for v in variants if v["category"] == "Language Instructions" and v["suite"] == "libero_goal"]
print("goal language examples:")
for v in goal_lang[:16] + goal_lang[-12:]:
    print(v)
print("other goal names:")
for cat in ["Robot Initial States", "Light Conditions", "Objects Layout"]:
    print(cat, [v for v in variants if v["suite"] == "libero_goal" and v["category"] == cat][:3])
print("VLM layer 0:", [(n, SHAPES[n]) for n in NAMES if "language_model.layers.0." in n])
print("expert layer 0:", [(n, SHAPES[n]) for n in NAMES if "gemma_expert.model.layers.0." in n])
```
