The earlier gates barely reduced the tagged-action error, and stronger gates sometimes made it worse. That leaves open both missed tag tokens and changes to the task-token embeddings themselves. I will finish reading the validation results and keep the architecture summary compact enough to inspect.
```python
print(prior_text[prior_text.find("### Four-detector gate rejected"):])
print("\nTENSORS:", len(NAMES), "EDITED:", len(edited()))
for n in NAMES:
    if ".blocks." not in n or ".blocks.0." in n:
        print(n, tuple(SHAPES[n]))
print("\nPRACTICE TYPE:", type(practice_raw).__name__)
if isinstance(practice_raw, dict):
    print("keys:", list(practice_raw))
else:
    print("count:", len(practice_raw), "examples:", practice_raw[:2])
print("\nWORK:", [p.name for p in W.iterdir()])
```
