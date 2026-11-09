I will check that the full-strength F edit is still installed and record the latest result. The remaining passes will focus on independent replication and mechanism checks, rather than tuning on the confirmation data.
```python
from pathlib import Path
import json, collections, hashlib, math
import numpy as np
import torch

W = Path(WORK)
print("ACTIVE EDITS:", edited())
print("ROOT FILES:")
for p in sorted(W.iterdir()):
    if p.is_dir():
        print(p.name + "/", "files=" + str(sum(1 for _ in p.iterdir())))
    elif not p.name.startswith("roll_"):
        print(p.name, p.stat().st_size)

r34 = [json.loads(x) for x in (W / "roll_34.jsonl").read_text().splitlines() if x.strip()]
print("\nPASS34 RECORD COUNT:", len(r34))
print("RECORD KEYS:", list(r34[0]))
print("EXAMPLE:", json.dumps(r34[0], ensure_ascii=True)[:1800])

for name in ["best_F_recipe.pt", "F_probe_mechanism_summary.json",
             "F_dose_probe_summary.json"]:
    p = W / name
    if not p.exists():
        continue
    if p.suffix == ".json":
        obj = json.loads(p.read_text())
        print("\n" + name + ":", json.dumps(obj, ensure_ascii=True)[:2200])
    else:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        print("\n" + name + " TYPE:", type(obj).__name__)
        if isinstance(obj, dict):
            print("KEYS:", list(obj)[:85])
            for k, v in list(obj.items())[:3]:
                print("EXAMPLE:", str(k), str(v)[:400])
        elif isinstance(obj, (list, tuple)):
            print("LENGTH:", len(obj), "FIRST:", str(obj[:3])[:1000])

entry = """
## Full-dose diagnostic replication (pass34)
Same80 diagnostic goal cases and seeds as half-dose pass33:
original49/80 -> full F68/80, +23.75pp SE4.76, 19 recoveries/0 regressions.
Half dose:59/80; full dose:68/80. This is development-set causal dose-response,
not independent confirmation. F remains the final candidate; no further tuning planned.
Restart at34/50 passes: audit artifacts, then replicate frozen F on independent
language-validation variants with fresh seeds, and strengthen fixed-input safety checks.
"""
notes = W / "notes.md"
if "Full-dose diagnostic replication (pass34)" not in notes.read_text():
    with notes.open("a") as f:
        f.write(entry)
print("\nBEST_F DIRECTORY:", [p.name for p in (W / "best_F_dev").iterdir()])
```
