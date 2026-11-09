I will first map the editable tensors and the practice variants, then reserve most of the forward-pass budget for paired causal tests and validation. I will keep a running log so that the final edit is supported by measured behavior, not just activation correlations.
I will inspect the model structure and practice-set coverage without reading any held-out data.
```python
import os, json, math, time, collections, pathlib, shutil
import numpy as np
import torch
torch.set_num_threads(min(24, os.cpu_count() or 1))
W = pathlib.Path(WORK)
B = pathlib.Path(BENCH)
variants_raw = json.loads((B / "practice_variants.json").read_text())
print("WORK", str(W))
print("variant container", type(variants_raw).__name__)
if isinstance(variants_raw, dict):
    print("keys", list(variants_raw)[:15])
else:
    print("count", len(variants_raw), "examples", variants_raw[:3])
print("editable tensors", len(NAMES))
for n in NAMES:
    if any(x in n for x in ["embed_tokens", "layers.0.", "action_", "time_mlp", "multi_modal_projector"]):
        print(n, SHAPES[n] if isinstance(SHAPES, dict) else SHAPES[NAMES.index(n)])
(W / "notes.md").write_text(
    "# Weight-edit investigation\n"
    "Goal: improve Language Instructions, preserving other categories. No training.\n"
    "Forward budget: 50. Plan: diagnose matched tasks; localize language-task features; "
    "test targeted edits at multiple doses; validate on disjoint practice variants and paired seeds; "
    "audit all other categories.\n"
    "Current best: original (no edits).\n"
)
```
 in? Tool output implicit next user? We emitted final code expected executes. Need wait.