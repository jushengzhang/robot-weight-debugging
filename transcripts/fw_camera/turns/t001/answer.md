I will first map the editable tensors and the available practice variants. This will let me choose matched camera and control tasks without accessing held-out data, and identify where a camera-specific repair could enter the shared attention pathway.```python
import os, json, math, time, re, collections
from pathlib import Path
import numpy as np
import torch

W = Path(WORK)
B = Path(BENCH)
torch.set_num_threads(min(32, os.cpu_count() or 8))
print("WORK:", str(W))
print("Editable tensors:", len(NAMES))
print("Existing edits:", list(edited()))
print("Tensor name/shape examples:")
for n in NAMES[:65]:
    print(n, tuple(SHAPES[n]) if isinstance(SHAPES, dict) else "")
with open(B / "practice_variants.json") as f:
    practice_raw = json.load(f)
print("Practice structure:", type(practice_raw).__name__)
if isinstance(practice_raw, dict):
    print("Top-level keys:", list(practice_raw)[:20])
else:
    print("Count:", len(practice_raw), "Examples:", practice_raw[:4])
(W / "notes.md").write_text(
    "# FastWAM camera robustness repair\n\n"
    "## Constraints and plan\n"
    "- Only direct, mechanism-specific weight edits; no training or local network execution.\n"
    "- Use practice variants only; reserve disjoint camera variants for confirmation.\n"
    "- Diagnose camera/control behavior and feature shifts, test causal doses, then paired validation.\n"
    "- Forward passes used: 0 / 40.\n"
    "- Current best: original weights (no edits).\n"
)
```
