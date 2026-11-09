I will recover the saved evaluation plan and verify that only the five frozen position-embedding rows differ from the checkpoint. Then I will continue the untouched-practice checks without further tuning.
```python
from pathlib import Path
import json, hashlib
import numpy as np
import torch

work = Path(WORK)
print("Non-rollout work files:")
for p in sorted(work.iterdir()):
    if not p.name.startswith(("roll_", "probe_")):
        print(p.name + ("/" if p.is_dir() else ""))

current_edits = list(edited())
print("\nEdited tensors:", current_edits)
for name in current_edits:
    x = load(name)
    x0 = load_original(name)
    changed = (x != x0)
    print(name, "shape", tuple(x.shape), "changed elements", int(changed.sum()))
    if x.ndim == 2 and x.shape[0] == 256:
        print("Changed rows:", torch.where(changed.any(dim=1))[0].tolist())
    print("Float32 byte hash:", hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest())

print("\nCandidate evaluation-plan files:")
for p in sorted(work.rglob("*.json")):
    if any(word in p.name.lower() for word in ("final", "safety", "freeze", "closing", "batch", "validation")):
        print(str(p.relative_to(work)))
```
