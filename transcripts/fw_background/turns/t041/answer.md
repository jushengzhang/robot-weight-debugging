I will reload the saved diagnostics and experiment manifests, then check how the fixed-input pairs were constructed. This will let me distinguish background-driven shifts from task or trajectory differences before choosing a targeted intervention.
```python
from pathlib import Path
import json, os, math, re, hashlib, time
import numpy as np
import torch
W = Path(WORK)
print("WORK:", W)
print("Files:", sorted(p.name for p in W.iterdir()))
print("Edited tensors:", edited())
for p in sorted(W.glob("*")):
    if p.is_dir() and (p.name.startswith("probe_") or p.name.startswith("roll_")):
        print(p.name, sorted(x.name for x in p.iterdir())[:12])
print("Notes tail:\n", (W / "notes.md").read_text()[-7000:])
print("Example names:", NAMES[:12])
```