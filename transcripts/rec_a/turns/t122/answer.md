Only expert layer 8, head 7 has been changed: its query block is multiplied by 1.5. I will use the same looping and successful observations from the earlier comparisons to check whether this sharpens spatial attention rather than merely amplifying an irrelevant image feature.
```python
print("Candidate:", (W / "current_candidate.json").read_text())
print("Forward-result directories:", [p.name for p in sorted((W / "fwd").iterdir())][-12:])
for k in [2, 3, 4, 31, 36, 37]:
    p = W / f"probe_{k}"
    if p.exists():
        print("\n", p.name, [x.name for x in p.iterdir()][:8])
        if (p/"index.json").exists():
            v=json.loads((p/"index.json").read_text())
            print(str(v)[:2500])
print("\nScreen episodes:", (W/"screen_episodes.json").read_text())
v=json.loads((W/"independent_panels.json").read_text())
print("Independent panel structure:", {k: (len(x) if hasattr(x,"__len__") else x) for k,x in v.items()} if isinstance(v,dict) else type(v))
```
