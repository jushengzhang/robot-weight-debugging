I will recover the saved audit panels and verify the installed edit. Neither frozen candidate improved pooled validation, so the remaining passes will check preservation and controls rather than tune another candidate.
```python
from pathlib import Path
import json
import numpy as np
import torch

W = Path(WORK)
print("Installed edits:", edited())
print("Saved panels/manifests:")
for p in sorted(W.iterdir()):
    if p.is_file() and p.suffix in (".json", ".md", ".py"):
        print(p.name, p.stat().st_size)
print("Candidate/audit directories:")
for p in sorted(W.iterdir()):
    if p.is_dir() and not p.name.startswith(("roll_", "probe_")):
        print(p.name, "files=", sum(1 for q in p.rglob("*") if q.is_file()))
print("Recent rollout files:", [p.name for p in sorted(
    W.glob("roll_*.jsonl"),
    key=lambda p: int(p.stem.split("_")[1])
)[-6:]])
```