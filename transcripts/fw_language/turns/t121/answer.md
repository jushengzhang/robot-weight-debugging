I will restore the saved evaluation plan, verify the active edit, and log pass 31. The candidate has not shown an independent language gain and has regressed on Robot Initial States, so I will finish the frozen evaluation without further tuning.
```python
from pathlib import Path
import json, math
import numpy as np
import torch

root = Path(WORK)
print("Active edits:", edited())
for name in edited():
    w, w0 = load(name), load_original(name)
    delta = w - w0
    print(name, "shape", tuple(w.shape),
          "changed", int(torch.count_nonzero(delta)),
          "relative_Frobenius_change", float(delta.norm() / w0.norm()))
    del w, w0, delta

print("\nRelevant saved files:")
for p in sorted(root.iterdir()):
    if p.is_file() and any(s in p.name for s in ("locked", "candidate", "notes", "comparison")):
        print(p.name, p.stat().st_size)

def brief(obj, depth=0):
    if isinstance(obj, dict):
        return {k: brief(v, depth + 1) for k, v in obj.items()} if depth < 2 else {"keys": list(obj)[:15]}
    if isinstance(obj, list):
        return {"length": len(obj), "first": brief(obj[0], depth + 1) if obj else None}
    return obj

for fn in ("locked_safety_evaluation_plan.json",
           "locked_language_validation_plan.json",
           "locked_evaluation_results.json"):
    data = json.loads((root / fn).read_text())
    print("\n" + fn)
    print(json.dumps(brief(data), indent=1))

rows31 = [json.loads(line) for line in (root / "roll_31.jsonl").read_text().splitlines() if line.strip()]
print("\nPass 31 record count:", len(rows31))
print("Pass 31 first-record structure:", json.dumps(brief(rows31[0]), indent=1)[:2500])
```
