I will first recover the saved diagnostics and check which edits are active. That will let me test the pending positional-encoding hypothesis without accidentally combining it with an earlier, rejected edit.
```python
from pathlib import Path
import json, numpy as np, torch, math, hashlib, os
W = Path(WORK)
print("WORK:", str(W))
print("Active edits:", edited())
print("Top-level files:")
for p in sorted(W.iterdir()):
    if p.is_dir():
        print(p.name + "/", "files=" + str(sum(1 for _ in p.iterdir())))
    else:
        print(p.name, p.stat().st_size)
print("\nNotes tail:")
print((W / "notes.md").read_text()[-5000:])
```
