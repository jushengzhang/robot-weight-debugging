I will first recover the saved edit recipes and recent paired results. The previous run ended on an assertion, so I need to verify the actual submitted weights before making another change.
```python
import os, json, pathlib, torch, numpy as np, collections, math
W = pathlib.Path(WORK)
print("WORK", str(W))
print("EDITED", list(edited()))
print("FILES")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    elif p.name.startswith(("best", "edit", "probe", "roll")):
        fs = list(p.iterdir())
        print(p.name + "/", len(fs), "entries", [q.name for q in fs[:4]])
print("\nNOTES")
print((W / "notes.md").read_text()[-16000:])
print("\nRECIPES")
for p in sorted(W.glob("*recipe*")):
    obj = torch.load(p, map_location="cpu", weights_only=False)
    print(p.name, type(obj).__name__)
    if isinstance(obj, dict):
        print(str(obj)[:6500])
```
