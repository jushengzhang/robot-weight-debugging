I’ll restore the saved experiment state and check the five-head query edit before testing it. The key question is whether softening these texture-sensitive attention heads improves control—not just whether it makes paired observations produce more similar actions.
```python
from pathlib import Path
import json, os, numpy as np, torch

W = Path(WORK)
torch.set_num_threads(min(16, os.cpu_count() or 1))

print("WORK:", W)
print("Current edited tensors:", edited())
print("\nTop-level saved artifacts:")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    else:
        print(p.name + "/")

print("\nQuery-normalization tensor names:")
for n in NAMES:
    if "action" in n and ("norm_q" in n or "q_norm" in n):
        print(n, SHAPES[n])

print("\nSaved Python utilities:")
for p in sorted(W.glob("*.py")):
    print("\n---", p.name, "---")
    print(p.read_text()[:4500])
```
