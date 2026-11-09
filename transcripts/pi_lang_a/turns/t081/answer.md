I will first check the saved files and active edits so I can resume the frozen 18-row candidate without changing it. Then I will finish the remaining safety check and run the precommitted second-seed language replication.
```python
from pathlib import Path
import json, math, collections
import numpy as np
import torch

W = Path(WORK)
print("Active edited tensors:", list(edited()))
print("Notes tail:")
print((W / "notes.md").read_text()[-6500:])

files = sorted(p.name for p in W.iterdir())
print("\nPlanning / audit / candidate files:")
print("\n".join(n for n in files if any(s in n.lower() for s in
    ["candidate", "replication", "repair", "safety", "canonical", "summary", "request", "alias"])
    and not n.startswith("roll_")))

for k in (38, 39):
    p = W / f"roll_{k}.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        print(f"\nPass {k}: {len(rows)} records")
        if rows:
            print("Record keys:", list(rows[0]))
            print("First record (excluding large fields):",
                  {a: b for a, b in rows[0].items()
                   if a in ["suite", "id", "seed", "model", "success", "termination", "instruction"]})

for fname in ["final18_replication_batches.json", "coverage_repair33.json",
              "canonical_replay_inputs.json"]:
    p = W / fname
    if p.exists():
        obj = json.loads(p.read_text())
        print("\n", fname, "type:", type(obj).__name__)
        if isinstance(obj, dict):
            print("Keys:", list(obj))
            print("Value lengths:", {k: len(v) for k, v in obj.items()
                                     if isinstance(v, (list, dict))})
        elif isinstance(obj, list):
            print("Length:", len(obj), "First entry:", str(obj[0])[:650] if obj else None)
```
