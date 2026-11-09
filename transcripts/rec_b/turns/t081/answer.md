The latest edit reduced recovery from 57/96 to 54/96 and increased loop failures, so I will reject it. I will first inspect the saved experiments and restore the original weights, then look for an upstream observation-processing problem rather than continue adjusting gripper-related pathways.
```python
import os, json, math, re, hashlib
from pathlib import Path
import numpy as np
import torch
from collections import Counter, defaultdict

W = Path(WORK)
print("Current edits:", edited())
for name in list(edited()):
    drop_edit(name)

print("Top-level files:")
for p in sorted(W.iterdir()):
    if not p.name.startswith(("roll_", "probe_")):
        print(p.name, "<dir>" if p.is_dir() else p.stat().st_size)
print("\nRecent rollout records:")
for k in range(29, 34):
    p = W / f"roll_{k}.jsonl"
    if p.exists():
        rows = [json.loads(s) for s in p.read_text().splitlines()]
        print(k, "rows", len(rows), "keys", list(rows[0]) if rows else [])
print("\nProbe directories:")
print([p.name for p in sorted(W.glob("probe_*"))])

with (W / "notes.md").open("a") as f:
    f.write("\n## Candidate K rejected, pass33\n"
            "Late action-only O17h6 attenuation:57/96->54/96,"
            "-3.1+/-3.1pp,3 rescues/6 regressions,loops35->36."
            "Current edits reset to NONE. 17 forward passes remain. "
            "Shift diagnosis upstream toward visual evidence and spatial localization; "
            "repeated expert-head interventions have not generalized.\n")
```
