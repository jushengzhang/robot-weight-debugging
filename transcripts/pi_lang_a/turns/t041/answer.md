I will reload the saved study records and inspect the last two passes to establish the current edit and what still needs independent validation.
```python
from pathlib import Path
import json, numpy as np, torch, re, collections, math
W = Path(WORK)
print("WORK:", str(W))
print("Edits:", edited())
print("Top-level files:")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    else:
        print(p.name + "/", len(list(p.iterdir())), "entries")
print("\nLast two probe/rollout records:")
for k in (15, 16):
    for p in (W/f"probe_{k}"/"results.json", W/f"roll_{k}.jsonl"):
        if p.exists():
            if p.suffix == ".json":
                obj = json.loads(p.read_text())
                text = json.dumps(obj, indent=1)
                print(str(p), text[:11500])
            else:
                rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
                print(str(p), "records:", len(rows), "first:", json.dumps(rows[0])[:4000])
print("\nNOTES:\n", (W/"notes.md").read_text())
```