I will first inventory the practice scenes and editable tensors, then collect a baseline with saved observations and activations. This will let me distinguish repeated-grasp failures from successful recoveries before choosing any weight changes.
```python
import os, json, math, collections, pathlib, time
import numpy as np
import torch
torch.set_grad_enabled(False)
W = pathlib.Path(WORK)
B = pathlib.Path(BENCH)
W.mkdir(exist_ok=True, parents=True)
with open(B / "practice_scenes.json") as f:
    practice_raw = json.load(f)
print("practice type:", type(practice_raw).__name__)
if isinstance(practice_raw, dict):
    print("top keys:", list(practice_raw)[:12])
    scenes = practice_raw.get("scenes", practice_raw.get("practice_scenes", practice_raw))
else:
    scenes = practice_raw
if isinstance(scenes, dict):
    scenes = list(scenes.values())
print("scenes:", len(scenes))
print("examples:", json.dumps(scenes[:3], ensure_ascii=True)[:3000])
for key in ["suite", "level", "stage"]:
    print(key, dict(collections.Counter(s.get(key, "?") for s in scenes)))
print("tensor count:", len(NAMES), "parameters:", sum(math.prod(SHAPES[n]) for n in NAMES))
print("non-block tensors:")
for n in NAMES:
    if ".layers." not in n and ".encoder.layers." not in n:
        print(n, SHAPES[n])
print("representative layer tensors:")
for n in NAMES:
    if ".layers.0." in n:
        print(n, SHAPES[n])
print("initial edits:", edited())
notes = """# Recovery weight-edit study
## Constraints and protocol
- No training, gradients, or kernel model execution; all inference via forward tool.
- Only listed practice scenes; paired original/edited evaluation.
- Diagnose loops using actions, images, activations, then causal weight interventions and dose-response.
- Reserve independent practice scenes for confirmation; ordinary-task evaluation availability to be checked.
## Status
- Inventory started; no edits, no forward passes used.
"""
(W / "notes.md").write_text(notes)
```
I will inventory the practice scenes and tensors first, then collect baseline trajectories with observations and activations. No weights will change until there is a concrete mechanism to test.

```python
import os, json, math, collections, pathlib, time
import numpy as np
import torch
torch.set_grad_enabled(False)
W = pathlib.Path(WORK)
B = pathlib.Path(BENCH)
W.mkdir(exist_ok=True, parents=True)
with open(B / "practice_scenes.json") as f:
    practice_raw = json.load(f)
print("practice type:", type(practice_raw).__name__)
if isinstance(practice_raw, dict):
    print("top keys:", list(practice_raw)[:12])
    scenes = practice_raw.get("scenes", practice_raw.get("practice_scenes", practice_raw))
else:
    scenes = practice_raw
if isinstance(scenes, dict):
    scenes = list(scenes.values())
print("scenes:", len(scenes))
print("examples:", json.dumps(scenes[:3], ensure_ascii=True)[:3000])
for key in ["suite", "level", "stage"]:
    print(key, dict(collections.Counter(s.get(key, "?") for s in scenes)))
print("tensor count:", len(NAMES), "parameters:", sum(math.prod(SHAPES[n]) for n in NAMES))
print("non-block tensors:")
for n in NAMES:
    if ".layers." not in n and ".encoder.layers." not in n:
        print(n, SHAPES[n])
print("representative layer tensors:")
for n in NAMES:
    if ".layers.0." in n:
        print(n, SHAPES[n])
print("initial edits:", edited())
(W / "notes.md").write_text("""# Recovery weight-edit study
## Protocol
No training, gradients, or kernel model execution. Only listed practice scenes.
Diagnose loops, perform targeted causal interventions with dose-response,
then confirm on independent practice scenes using paired original/edited trials.
## Status
Inventory started; no edits and no forward passes used.
""")
```
