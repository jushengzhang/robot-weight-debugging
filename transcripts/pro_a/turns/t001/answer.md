I will first inventory the practice episodes and editable tensors, then choose matched original/perturbed episodes for diagnosis. I will keep a separate validation set untouched while choosing edits, and record all experiments in `WORK/notes.md`.
```python
import os, json, math, pathlib, collections, time, numpy as np, torch
W = pathlib.Path(WORK)
B = pathlib.Path(BENCH)
torch.set_num_threads(min(24, os.cpu_count() or 1))
raw_episodes = json.loads((B / "practice_episodes.json").read_text())
print("Episode file type:", type(raw_episodes).__name__)
if isinstance(raw_episodes, dict):
    print("Top-level keys:", list(raw_episodes)[:20])
    episodes = raw_episodes.get("episodes", raw_episodes.get("practice_episodes"))
else:
    episodes = raw_episodes
print("Episode count:", len(episodes))
print("First three:", episodes[:3])
print("Types:", dict(collections.Counter(e["ptype"] for e in episodes)))
print("Suites:", dict(collections.Counter(e["suite"] for e in episodes)))
print("Editable tensors:", len(NAMES))
print("Key tensors:")
for n in NAMES:
    if (".layers.0." in n or "position_embedding" in n or
        "multi_modal_projector" in n or n.startswith(("action_", "time_"))):
        print(n, SHAPES[n] if isinstance(SHAPES, dict) else SHAPES[NAMES.index(n)])
notes = """# Robustness weight-edit study
Constraints: direct, mechanistically motivated weight edits only; no training, no local model execution.
Forward budget: 40. Starting with matched diagnosis and a predefined practice validation partition.
All comparisons will use fixed episode seeds; final checks include original/object/semantic.
"""
(W / "notes.md").write_text(notes)
```
```python
import os, json, math, pathlib, collections, time, numpy as np, torch
W = pathlib.Path(WORK)
B = pathlib.Path(BENCH)
torch.set_num_threads(min(24, os.cpu_count() or 1))
raw_episodes = json.loads((B / "practice_episodes.json").read_text())
print("Episode file type:", type(raw_episodes).__name__)
if isinstance(raw_episodes, dict):
    print("Top-level keys:", list(raw_episodes)[:20])
    episodes = raw_episodes.get("episodes", raw_episodes.get("practice_episodes"))
else:
    episodes = raw_episodes
print("Episode count:", len(episodes))
print("First three:", episodes[:3])
print("Types:", dict(collections.Counter(e["ptype"] for e in episodes)))
print("Suites:", dict(collections.Counter(e["suite"] for e in episodes)))
print("Editable tensors:", len(NAMES))
print("Key tensors:")
for n in NAMES:
    if (".layers.0." in n or "position_embedding" in n or
        "multi_modal_projector" in n or n.startswith(("action_", "time_"))):
        print(n, SHAPES[n] if isinstance(SHAPES, dict) else SHAPES[NAMES.index(n)])
notes = """# Robustness weight-edit study
Constraints: direct, mechanistically motivated weight edits only; no training, no local model execution.
Forward budget: 40. Starting with matched diagnosis and a predefined practice validation partition.
All comparisons will use fixed episode seeds; final checks include original/object/semantic.
"""
(W / "notes.md").write_text(notes)
```