I will reload the saved experiment records and current edit, then check the visual-aperture intervention in paired rollouts. The small action changes in the last probe are not enough to tell whether it helps recovery.
```python
from pathlib import Path
import json, numpy as np, torch
from collections import Counter, defaultdict
W = Path(WORK)
print("WORK", str(W))
print("files", sorted(p.name for p in W.iterdir()))
print("edited", edited())
for fn in ["splits.json", "visual_grip_direction.pt"]:
    p = W / fn
    if p.exists():
        if p.suffix == ".json":
            obj = json.loads(p.read_text())
            print(fn, "type", type(obj).__name__)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    print(" ", k, "len", len(v) if hasattr(v, "__len__") else "-", "sample", str(v)[:500])
        else:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            print(fn, {k: (tuple(v.shape) if torch.is_tensor(v) else str(v)[:150])
                       for k, v in obj.items()} if isinstance(obj, dict) else type(obj))
for fn in ["roll_13.jsonl", "probe_16/index.json"]:
    p = W / fn
    if p.exists():
        if p.suffix == ".jsonl":
            rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
            print(fn, "n", len(rows), "first", str(rows[0])[:1800])
        else:
            print(fn, p.read_text()[:2500])
```
