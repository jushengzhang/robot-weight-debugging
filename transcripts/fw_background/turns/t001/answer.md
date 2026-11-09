I will first map the editable tensors and practice variants, then collect matched background/control rollouts. I will keep diagnosis, candidate edits, and validation results in `WORK/notes.md`, and reserve passes for paired tests on practice variants not used to select the edit.

I will check the parameter layout before choosing which activations to record, so the first diagnostic rollouts target the relevant video and action components.

The initial inspection will only list tensor shapes and variant counts; it will not change any weights.

```python
import os, json, re, math, time, random, hashlib, shutil
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch

W = Path(WORK)
B = Path(BENCH)
torch.set_num_threads(min(16, os.cpu_count() or 8))
print("WORK:", str(W))
print("Editable tensors:", len(NAMES), "existing edits:", len(edited()))
print("Top-level prefixes:", Counter(n.split(".")[0] for n in NAMES))
for n in NAMES:
    if not re.search(r"(?:blocks|layers)\.(?:[1-9]|[12][0-9])\.", n):
        print(n, tuple(SHAPES[n]))
raw_variants = json.loads((B / "practice_variants.json").read_text())
print("Variant JSON type:", type(raw_variants).__name__)
if isinstance(raw_variants, dict):
    print("JSON keys:", list(raw_variants)[:20])
else:
    print("First variants:", raw_variants[:3])
(W / "notes.md").write_text(
    "# FastWAM background robustness\n\n"
    "## Rules and plan\n"
    "- No training, network execution in Python, or non-practice benchmark access.\n"
    "- Diagnose matched background/control behavior and internal changes.\n"
    "- Use targeted, non-uniform weight edits with causal dose-response tests.\n"
    "- Reserve disjoint practice variants for paired background and other-category validation.\n"
    "- Forward passes used: 0 / 40. No edits yet.\n"
)
```
