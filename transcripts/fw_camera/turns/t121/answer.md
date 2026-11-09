I will recover the saved evaluation panels and verify that the frozen candidate is unchanged. The last three passes are reserved for a Sensor Noise replication and two preselected camera panels, with no further weight tuning.
```python
from pathlib import Path
import json, hashlib
import numpy as np
import torch

W = Path(WORK)
active_names = list(edited())
print("Active edits:", active_names)
for name in active_names:
    t = load(name).detach().cpu().contiguous()
    print(name, tuple(t.shape),
          "sha256:", hashlib.sha256(t.numpy().tobytes()).hexdigest())

def compact(obj, depth=0):
    if isinstance(obj, dict):
        return {k: compact(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 8:
            return {"length": len(obj), "first": compact(obj[0], depth + 1),
                    "last": compact(obj[-1], depth + 1)}
        return [compact(v, depth + 1) for v in obj]
    return obj

plan_files = [
    p for p in sorted(W.glob("*.json"))
    if any(s in p.name.lower() for s in
           ("final", "sensor", "candidate_lock", "safety", "remaining"))
]
print("Planning artifacts:", [p.name for p in plan_files])
for p in plan_files:
    data = json.loads(p.read_text())
    text = json.dumps(compact(data), ensure_ascii=True)
    print(p.name, text[:5000])

p = W / "roll_36.jsonl"
if p.exists():
    with p.open() as f:
        print("Rollout record schema/example:", f.readline().strip()[:1800])
```
