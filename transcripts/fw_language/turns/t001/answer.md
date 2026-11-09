I will first inspect the editable modules and practice-variant metadata, then establish a paired baseline. I will focus on the text-conditioning path, but will not choose an edit until the rollout and probe evidence identifies a specific failure mechanism.
I will also create a running notes file so that the baseline, causal tests, and final validation remain separate and auditable.
```python
import os, json, math, time, re, collections
from pathlib import Path
import numpy as np
import torch

W = Path(WORK)
B = Path(BENCH)
torch.set_num_threads(min(16, os.cpu_count() or 1))
print("Editable tensors:", len(NAMES))
print("WORK entries:", [p.name for p in W.iterdir()])
print("\nRepresentative names/shapes:")
for n in NAMES[:65]:
    print(n, tuple(SHAPES[n]))
print("\nOther module prefixes:")
print(collections.Counter(".".join(n.split(".")[:2]) for n in NAMES))
with open(B / "practice_variants.json") as f:
    practice_raw = json.load(f)
print("\nPractice metadata type:", type(practice_raw).__name__)
if isinstance(practice_raw, list):
    variants = practice_raw
    print("Count:", len(variants), "Examples:", variants[:6])
else:
    print("Top-level keys:", list(practice_raw)[:20])
    variants = practice_raw.get("variants", practice_raw.get("practice_variants", []))
    print("Examples:", variants[:6] if isinstance(variants, list) else str(variants)[:1500])
notes = """# FastWAM language-robustness weight-edit investigation

## Constraints and plan
- No training, backward passes, or policy execution in the Python kernel.
- Only practice variants and forward-tool outputs are inspected.
- Diagnose language failures against less perturbed variants of the same tasks.
- Test localized edits, include causal dose-response and untouched validation.
- Require paired language gains larger than paired uncertainty and check all other categories.
- Forward budget: 40 passes. No passes used yet.

## Current submitted edit
None.

"""
(W / "notes.md").write_text(notes)
print("\nNotes initialized:", str(W / "notes.md"))
```
