I will restore the saved experiment state and inspect the latest candidate. The last result, +1.6 points with a 2.9-point standard error, is inconclusive, so I will not treat that edit as an improvement yet.
```python
import os, json, math, hashlib
from pathlib import Path
import numpy as np
import torch

W = Path(WORK)
print("WORK:", W)
print("Edited tensors:", edited())
print("Top-level artifacts:")
for p in sorted(W.iterdir()):
    if p.is_dir():
        print(" DIR ", p.name, "entries=", len(list(p.iterdir())))
    else:
        print(" FILE", p.name, "bytes=", p.stat().st_size)
print("\nNotes tail:")
notes = W / "notes.md"
print(notes.read_text()[-11000:] if notes.exists() else "(missing)")
print("\nRollout record examples:")
for k in (4, 5):
    p = W / f"roll_{k}.jsonl"
    if p.exists():
        rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        print("roll", k, "rows", len(rows))
        for r in rows[:2]:
            print(json.dumps(r)[:1800])
print("\nRelevant normalization names:")
print([n for n in NAMES if "29" in n and ("norm_q" in n or "norm_k" in n)])
```
