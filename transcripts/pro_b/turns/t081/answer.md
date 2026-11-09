I will reload the saved diagnostics and edit state before choosing the next intervention. The earlier tests ruled out several simple attention-strength explanations; the remaining passes need to focus on a narrower mechanism and leave room for independent paired validation.
```python
from pathlib import Path
import json, numpy as np, torch, math, hashlib, collections
W = Path(WORK)
torch.set_num_threads(16)
print("Current edits:", edited())
print("WORK files:")
for p in sorted(W.iterdir()):
    if p.is_dir():
        print(p.name + "/", len(list(p.iterdir())), "entries")
    else:
        print(p.name, p.stat().st_size)
print("\nRecent notes:")
print((W / "notes.md").read_text()[-7000:])
```
