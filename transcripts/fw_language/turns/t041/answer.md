I will inspect the saved probes and analysis files to recover the exact instruction pairs and available activations. This will let me continue the diagnosis without repeating passes or relying on the static weight outliers alone.
```python
from pathlib import Path
import os, json, numpy as np, torch
W = Path(WORK)
print("WORK:", W)
print("Active edits:", edited())
print("Files:")
for p in sorted(W.iterdir()):
    if p.is_dir():
        print(p.name + "/", "entries", len(list(p.iterdir())))
    else:
        print(p.name, p.stat().st_size)
print("\nTensor names (sample):", NAMES[:18])
print("\nProbe directories:")
for p in sorted(W.glob("probe_*")):
    print(p.name, [q.name for q in list(p.iterdir())[:12]])
```
