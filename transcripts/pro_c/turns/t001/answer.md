I will first inventory the weights and practice episodes, then choose matched original/perturbed cases for diagnosis. I will reserve separate episodes for validation so that an edit is not judged only on the cases that suggested it.
I will also create a running notes file and record a pass budget: diagnosis first, causal interventions next, and paired validation before choosing the final weights.
The first inspection only reads the permitted episode list and weight metadata; it does not run the network.
```python
import os, json, math, random, time, collections, pathlib
import numpy as np
import torch
torch.set_num_threads(16)
W = pathlib.Path(WORK)
B = pathlib.Path(BENCH)
raw_eps = json.loads((B / "practice_episodes.json").read_text())
print("episode container:", type(raw_eps).__name__)
if isinstance(raw_eps, dict):
    print("keys:", list(raw_eps)[:20])
eps = raw_eps if isinstance(raw_eps, list) else raw_eps.get("episodes", raw_eps.get("practice_episodes"))
print("n episodes:", len(eps), "n tensors:", len(NAMES))
print("first episodes:", eps[:3])
print("counts:", dict(collections.Counter(e["ptype"] for e in eps)))
print("suites:", dict(collections.Counter(e["suite"] for e in eps)))
print("representative tensor names/shapes:")
for n in NAMES:
    if any(x in n for x in ("layers.0.", "action_", "time_mlp", "multi_modal_projector")) or n.endswith("norm.weight"):
        print(n, SHAPES[n])
notes = """# Weight-only robustness repair
## Constraints and plan
No training, no local network execution, practice episodes only.
40-pass plan: ~6 diagnostic, ~20 causal/dose-response, ~14 validation/refinement.
Evaluate paired original/edited with fixed seeds; preserve original/object/semantic performance.
Initial edits: none.
"""
(W / "notes.md").write_text(notes)
print("WORK:", str(W))
``` 
```python
import os, json, math, random, time, collections, pathlib
import numpy as np
import torch
torch.set_num_threads(16)
W = pathlib.Path(WORK)
B = pathlib.Path(BENCH)
raw_eps = json.loads((B / "practice_episodes.json").read_text())
print("episode container:", type(raw_eps).__name__)
if isinstance(raw_eps, dict):
    print("keys:", list(raw_eps)[:20])
eps = raw_eps if isinstance(raw_eps, list) else raw_eps.get("episodes", raw_eps.get("practice_episodes"))
print("n episodes:", len(eps), "n tensors:", len(NAMES))
print("first episodes:", eps[:3])
print("counts:", dict(collections.Counter(e["ptype"] for e in eps)))
print("suites:", dict(collections.Counter(e["suite"] for e in eps)))
print("representative tensor names/shapes:")
for n in NAMES:
    if any(x in n for x in ("layers.0.", "action_", "time_mlp", "multi_modal_projector")) or n.endswith("norm.weight"):
        print(n, SHAPES[n])
notes = """# Weight-only robustness repair
## Constraints and plan
No training, no local network execution, practice episodes only.
40-pass plan: ~6 diagnostic, ~20 causal/dose-response, ~14 validation/refinement.
Evaluate paired original/edited with fixed seeds; preserve original/object/semantic performance.
Initial edits: none.
"""
(W / "notes.md").write_text(notes)
print("WORK:", str(W))
```
