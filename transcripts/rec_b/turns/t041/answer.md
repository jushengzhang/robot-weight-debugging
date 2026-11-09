I will restore the experiment state from `/work`, verify the pending edit, and inspect the saved evidence. That will let me resume the causal tests without repeating passes or mistaking a small trajectory change for a reliable recovery gain.
```python
import os, json, pathlib, numpy as np, torch, re
from collections import Counter, defaultdict
W = pathlib.Path(WORK)
print("Current edits:", edited())
print("WORK files:")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    else:
        print(p.name + "/", len(list(p.iterdir())), "entries")
print("\nNotes tail:\n", (W / "notes.md").read_text()[-3500:])
for n in edited():
    a, b = load(n), load_original(n)
    d = a - b
    nz = (d != 0)
    print("EDIT", n, "shape", tuple(a.shape),
          "changed", int(nz.sum()), "relative_norm", float(d.norm()/b.norm()))
    if a.ndim == 2 and "self_attn.o_proj" in n:
        print("head relative changes",
              [round(float(d[:, 256*h:256*(h+1)].norm() /
                           b[:, 256*h:256*(h+1)].norm()), 4) for h in range(8)])
    del a, b, d, nz
```
