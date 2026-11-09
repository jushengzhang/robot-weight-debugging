I’ll read the saved notes and probe results to recover the experiment state, then check whether the temporary norm-only control reproduces the repair. This will help distinguish a semantic change in the edited embeddings from a change in their magnitudes.
```python
from pathlib import Path
import json, numpy as np, torch, gc

W = Path(WORK)
print("WORK:", str(W))
print("Current edits:", edited())
print("Top-level files:", sorted(p.name for p in W.iterdir()))
print("Causal controls:", sorted(p.name for p in (W/"causal_controls").iterdir()))
print("Final checkpoint:", sorted(p.name for p in (W/"checkpoints"/"lex_dest_exact").iterdir()))

for k in [22, 23, 38]:
    p = W/f"probe_{k}"/"results.json"
    obj = json.loads(p.read_text())
    print("\nPROBE", k, "type:", type(obj).__name__)
    if isinstance(obj, dict):
        print("Keys:", list(obj))
        for key, val in obj.items():
            if isinstance(val, list):
                print(key, "length:", len(val), "first:", str(val[:1])[:2300])
            else:
                print(key, str(val)[:700])
    else:
        print("Length:", len(obj), "first:", str(obj[:1])[:2300])

small_metadata = [
    p for p in W.rglob("*.json")
    if ("causal" in str(p) or "lex_dest_exact" in str(p)) and p.stat().st_size < 20000
]
for p in small_metadata:
    print("\nMETADATA", p.relative_to(W), p.read_text()[:3500])
```
