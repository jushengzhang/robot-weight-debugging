I will reload the saved analysis and inspect the probe files to resume the gripper-path diagnosis without repeating passes. I will also update the notes’ stale pass count and confirm that no rejected edits remain active.
```python
from pathlib import Path
import json, os, numpy as np, torch, math
W = Path(WORK)
torch.set_num_threads(12)

print("Active edits:", edited())
print("Top-level WORK files:")
for p in sorted(W.iterdir()):
    if p.is_dir():
        print(p.name + "/", "items", len(list(p.iterdir())))
    else:
        print(p.name, p.stat().st_size)

notes = (W / "notes.md").read_text()
notes = notes.replace("Forward passes used: 0/50.", "Forward passes used: 14/50; code runs before restart: 26/300.")
(W / "notes.md").write_text(notes)

for name in ("scene_split.json",):
    p = W / name
    if p.exists():
        x = json.loads(p.read_text())
        print(name, str(x)[:2200])

for k in (4, 12, 13, 14):
    d = W / f"probe_{k}"
    if d.exists():
        print("PROBE", k, [(p.name, p.stat().st_size) for p in sorted(d.iterdir())[:12]])
        for p in sorted(d.glob("*.json"))[:1]:
            print(p.name, p.read_text()[:2800])

print("Saved analysis scripts:", [p.name for p in W.glob("*.py")])
```
