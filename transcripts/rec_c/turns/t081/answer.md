I will restore the experiment state from `/work` and check the active edit and reserved validation episodes. The visual gripper-cue gain helped slightly on development scenes, but I need the untouched paired test before treating it as an improvement.
```python
from pathlib import Path
import json, numpy as np, torch, math, collections
torch.set_num_threads(16)
W = Path(WORK)
print("WORK", str(W))
print("active edits", edited())
print("top-level files", sorted(p.name for p in W.iterdir() if p.is_file()))
v2 = json.loads((W / "validation_batch2.json").read_text())
print("validation2 structure", type(v2).__name__)
if isinstance(v2, dict):
    print("keys", list(v2))
    for k, v in v2.items():
        print(k, str(v)[:700])
else:
    print("count", len(v2), "first entries", v2[:4])
for p in sorted(W.glob("*.py")):
    print("script", p.name, p.stat().st_size)
print("recent rollouts", sorted(p.name for p in W.glob("roll_*.jsonl"))[-8:])
print("candidate directories", sorted(p.name for p in W.iterdir()
      if p.is_dir() and not p.name.startswith(("roll_", "probe_"))))
```
