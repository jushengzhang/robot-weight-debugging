I will recover the current edit and experiment history from `/work`, especially the changes tested in passes 16–17. Then I will verify the candidate’s mechanism, freeze it before opening the reserved validation results, and check the other perturbation categories for regressions.
```python
from pathlib import Path
import json, os, numpy as np, torch
W = Path(WORK)
print("WORK:", W)
print("Edited tensors:", edited())
print("\nTop-level artifacts:")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    else:
        print(p.name + "/")
print("\nNotes:")
print((W / "notes.md").read_text()[-10000:])
print("\nCheckpoints:")
cp = W / "checkpoints"
if cp.exists():
    for p in sorted(cp.iterdir()):
        print(p.name, [q.name for q in p.iterdir()] if p.is_dir() else p.stat().st_size)
for k in (16, 17):
    p = W / f"roll_{k}.jsonl"
    if p.exists():
        rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        print(f"\nroll_{k}: n_records={len(rows)}, first record keys={list(rows[0])}")
        print(json.dumps(rows[0], ensure_ascii=True)[:2500])
```
