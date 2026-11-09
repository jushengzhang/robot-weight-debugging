I will first inventory the practice episodes and editable tensors, then reserve a validation split before choosing any edits. I’ll focus diagnosis on task, position, and environment failures, while keeping original, object, and semantic episodes as regression checks.
```python
import os, json, math, random, hashlib, collections, time
from pathlib import Path
import numpy as np
import torch

W = Path(WORK)
B = Path(BENCH)
torch.set_num_threads(min(32, os.cpu_count() or 8))
episodes_raw = json.loads((B / "practice_episodes.json").read_text())
print("practice format:", type(episodes_raw).__name__)
if isinstance(episodes_raw, dict):
    print("top-level keys:", list(episodes_raw)[:12])
    episodes = episodes_raw.get("episodes", episodes_raw.get("practice_episodes", []))
else:
    episodes = episodes_raw
print("episodes:", len(episodes))
print("ptype counts:", dict(collections.Counter(e["ptype"] for e in episodes)))
print("suite counts:", dict(collections.Counter(e["suite"] for e in episodes)))
print("example:", episodes[:2])
print("tensor count:", len(NAMES), "parameters:", sum(math.prod(SHAPES[n]) for n in NAMES))
for stem in ("vision_model.encoder.layers.0", "language_model.layers.0", "gemma_expert.model.layers.0"):
    print("\n", stem)
    for n in NAMES:
        if stem in n:
            print(n, SHAPES[n])
print("\nnon-layer tensors:")
for n in NAMES:
    if ".layers." not in n and ".encoder.layers." not in n:
        print(n, SHAPES[n])
notes = """# Weight-only robustness repair
## Constraints and plan
- No training or local network execution. Only direct, mechanistic weight edits.
- 40 forward passes available. Start with paired diagnostic observations, then causal interventions and dose-response, then disjoint validation.
- Main hypotheses: scene-habit directions override instruction; fixed-position features impede moved-object localization; appearance-sensitive visual modes impede environment transfer.
- No edit chosen yet.
"""
(W / "notes.md").write_text(notes)
```
I will inventory the practice episodes and editable tensors first. This will let me choose matched diagnostic cases and reserve separate validation episodes before testing edits.

```python
import os, json, math, random, hashlib, collections, time
from pathlib import Path
import numpy as np
import torch

W = Path(WORK)
B = Path(BENCH)
torch.set_num_threads(min(32, os.cpu_count() or 8))
episodes_raw = json.loads((B / "practice_episodes.json").read_text())
print("practice format:", type(episodes_raw).__name__)
if isinstance(episodes_raw, dict):
    print("top-level keys:", list(episodes_raw)[:12])
    episodes = episodes_raw.get("episodes", episodes_raw.get("practice_episodes", []))
else:
    episodes = episodes_raw
print("episodes:", len(episodes))
print("ptype counts:", dict(collections.Counter(e["ptype"] for e in episodes)))
print("suite counts:", dict(collections.Counter(e["suite"] for e in episodes)))
print("example:", episodes[:2])
print("tensor count:", len(NAMES), "parameters:", sum(math.prod(SHAPES[n]) for n in NAMES))
for stem in ("vision_model.encoder.layers.0", "language_model.layers.0", "gemma_expert.model.layers.0"):
    print("\n", stem)
    for n in NAMES:
        if stem in n:
            print(n, SHAPES[n])
print("\nnon-layer tensors:")
for n in NAMES:
    if ".layers." not in n and ".encoder.layers." not in n:
        print(n, SHAPES[n])
notes = """# Weight-only robustness repair
## Constraints and plan
- No training or local network execution. Only direct, mechanistic weight edits.
- 40 forward passes available. Start with paired diagnostic observations, then causal interventions and dose-response, then disjoint validation.
- Main hypotheses: scene-habit directions override instruction; fixed-position features impede moved-object localization; appearance-sensitive visual modes impede environment transfer.
- No edit chosen yet.
"""
(W / "notes.md").write_text(notes)
```
