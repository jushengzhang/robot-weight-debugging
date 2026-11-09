I will recover the saved analysis and inspect the latest text-context readout. The current edits are diagnostic instrumentation, not a repair; I will remove them before testing any policy changes.
```python
from pathlib import Path
import json, numpy as np, torch, os
W = Path(WORK)
print("WORK:", W)
print("Top-level files:")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    elif not p.name.startswith(("roll_", "probe_")):
        print(p.name + "/")
print("\nAnalysis/code files:")
for p in sorted(W.rglob("*")):
    if p.is_file() and p.suffix in (".py", ".md") and "edited" not in str(p):
        print(str(p.relative_to(W)), p.stat().st_size)
print("\nCurrent edits:", edited())
p = W / "probe_15" / "results.json"
r15 = json.loads(p.read_text())
print("\nProbe 15 structure:", str(r15)[:3500])
```
