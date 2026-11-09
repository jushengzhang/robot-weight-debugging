I will recover the saved edit, probe metadata, and development episode list so the next test uses the same paired seeds. The current candidate amplifies an instruction-contrast subspace in four VLM value projections; it still needs a causal test.
```python
import os, json, math, pathlib, numpy as np, torch
from collections import Counter, defaultdict
torch.set_num_threads(16)
W = pathlib.Path(WORK)
print("WORK", W)
print("FILES", sorted(p.name for p in W.iterdir()))
print("EDITED", edited())
for name in edited():
    a, b = load(name), load_original(name)
    print(name, tuple(a.shape), "relative_change", round(float((a-b).norm()/b.norm()), 5))
for p in sorted(W.glob("roll_*.jsonl")):
    rows = [json.loads(s) for s in p.read_text().splitlines()]
    print(p.name, len(rows), "keys", list(rows[0]) if rows else [])
print("metadata files", [p.name for p in W.glob("*.json")])
```
