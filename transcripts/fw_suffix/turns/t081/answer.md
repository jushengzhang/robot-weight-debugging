I’ll inspect the saved diagnostics and current edits to recover the exact state. The latest 96-episode test showed no net gain, so I’ll reserve the remaining passes for a targeted alternative and independent tagged, clean, and paraphrase checks.
```python
import os, json, math, pathlib, torch, numpy as np
from collections import Counter, defaultdict
W = pathlib.Path(WORK)
torch.set_num_threads(12)
print("WORK:", str(W))
print("Edited:", len(edited()), list(edited())[:12])
print("Root files:")
for p in sorted(W.iterdir()):
    if not p.name.startswith(("roll_", "probe_")):
        print(p.name, "<dir>" if p.is_dir() else p.stat().st_size)
print("Python/metadata files:")
for p in sorted(W.glob("*.py")):
    print("\nFILE", p.name)
    print(p.read_text()[:14000])
print("Text/cross-attention tensor examples:")
sel = [n for n in NAMES if any(s in n for s in ("text_embedding", "cross_attn", "proprio"))]
for n in sel[:40]:
    print(n, SHAPES[n])
print("Total tensors:", len(NAMES))
```
