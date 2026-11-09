I will first inventory the practice variants and editable tensors, then select matched literal/paraphrased tasks for diagnosis. I will keep a running record of hypotheses, causal tests, and paired evaluation results in `WORK/notes.md`.
The first inspection will only read the supplied variant list and tensor metadata; it will not run the network or change any weights.
```python
import os, json, math, time, collections, pathlib
import numpy as np
import torch
torch.set_num_threads(16)
W = pathlib.Path(WORK)
B = pathlib.Path(BENCH)
variants = json.loads((B / "practice_variants.json").read_text())
print("WORK:", str(W))
print("Variant container:", type(variants).__name__)
if isinstance(variants, dict):
    print("Keys:", list(variants)[:15])
else:
    print("Count:", len(variants), "First records:", variants[:4])
print("Tensor count:", len(NAMES))
print("Representative tensors:", [(n, SHAPES[n] if isinstance(SHAPES, dict) else SHAPES[i])
                                   for i, n in enumerate(NAMES)
                                   if any(s in n for s in ["embed_tokens", "layers.0.self_attn", "action_in_proj"])][:20])
(W / "notes.md").write_text(
    "# Language robustness weight-edit experiment\n"
    "## Constraints\nNo training, no kernel network execution; practice variants only. "
    "Use paired rollouts and independent validation. Budget: 50 forward passes.\n"
    "## Current status\nNo edits. Inventory and matched-task diagnosis starting.\n")
```
```python
import os, json, math, time, collections, pathlib
import numpy as np
import torch
torch.set_num_threads(16)
W = pathlib.Path(WORK)
B = pathlib.Path(BENCH)
variants = json.loads((B / "practice_variants.json").read_text())
print("WORK:", str(W))
print("Variant container:", type(variants).__name__)
if isinstance(variants, dict):
    print("Keys:", list(variants)[:15])
else:
    print("Count:", len(variants), "First records:", variants[:4])
print("Tensor count:", len(NAMES))
print("Representative tensors:", [(n, SHAPES[n] if isinstance(SHAPES, dict) else SHAPES[i])
                                   for i, n in enumerate(NAMES)
                                   if any(s in n for s in ["embed_tokens", "layers.0.self_attn", "action_in_proj"])][:20])
(W / "notes.md").write_text(
    "# Language robustness weight-edit experiment\n"
    "## Constraints\nNo training, no kernel network execution; practice variants only. "
    "Use paired rollouts and independent validation. Budget: 50 forward passes.\n"
    "## Current status\nNo edits. Inventory and matched-task diagnosis starting.\n")
```
