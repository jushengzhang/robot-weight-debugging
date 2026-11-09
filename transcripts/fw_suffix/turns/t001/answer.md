I will first inspect the editable tensors and the practice-variant list, then set up a log and fixed diagnostic splits. This will let me target the text-to-action pathways and reserve separate variants for validation.
```python
import os, json, math, time, re, collections
from pathlib import Path
import numpy as np
import torch

work = Path(WORK)
bench = Path(BENCH)
torch.set_num_threads(min(32, os.cpu_count() or 1))
variants = json.loads((bench / "practice_variants.json").read_text())
print("WORK:", str(work))
print("Editable tensors:", len(NAMES))
print("Variant container:", type(variants).__name__)
if isinstance(variants, dict):
    print("Variant keys:", list(variants)[:20])
    for k, v in variants.items():
        if isinstance(v, list):
            print(k, "count:", len(v), "example:", v[:2])
else:
    print("Practice variants:", len(variants), "examples:", variants[:3])
print("\nTensor names/shapes (representative):")
representative = [n for n in NAMES
                  if not re.search(r"(?:blocks|layers)\.\d+\.", n)
                  or re.search(r"(?:blocks|layers)\.0\.", n)]
for n in representative[:200]:
    print(n, tuple(SHAPES[n]))
(work / "notes.md").write_text(
    "# FastWAM tag-invariance weight edit\n\n"
    "## Constraints\n"
    "Direct weight analysis/edits only; no training or local model execution. "
    "Use only listed practice variants; preserve clean and paraphrased language.\n\n"
    "## Plan\n"
    "1. Inspect architecture/weights and establish matched raw/strip diagnostics.\n"
    "2. Localize tag influence in action/video cross-attention, test causal doses.\n"
    "3. Validate a chosen edit on reserved practice variants with paired original/edited "
    "rollouts; test strip and Language Instructions separately.\n\n"
    "## Pass log\n"
    "No forward passes used yet.\n\n"
    "## Current edit\nNone.\n"
)
```
