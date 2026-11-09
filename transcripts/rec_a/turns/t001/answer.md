I will first inventory the weights and practice scenes, then reserve scenes for validation before inspecting any outcomes. I will use paired rollouts and targeted probes to distinguish a stale grasp prior from poor visual localization or a gripper-output defect, and keep a running record of the evidence and edits.
The first inspection will only read the permitted scene list and weight metadata. It will also create a notes file and a reproducible scene-level discovery/validation split so that later gains can be tested on scenes not used to choose an edit.
```python
import os, json, math, hashlib, collections, time
from pathlib import Path
import numpy as np
import torch

W = Path(WORK)
B = Path(BENCH)
W.mkdir(exist_ok=True, parents=True)
scenes_raw = json.loads((B / "practice_scenes.json").read_text())
print("Scene file type:", type(scenes_raw).__name__)
if isinstance(scenes_raw, dict):
    print("Top-level keys:", list(scenes_raw)[:20])
    scenes = scenes_raw.get("scenes", scenes_raw.get("practice_scenes", scenes_raw))
else:
    scenes = scenes_raw
print("Scene count:", len(scenes))
print("First scene:", scenes[0] if isinstance(scenes, list) else list(scenes.items())[:1])
print("Tensors:", len(NAMES), "parameters:", sum(math.prod(SHAPES[n]) for n in NAMES))
print("Non-block tensors:")
for n in NAMES:
    if ".layers." not in n:
        print(n, tuple(SHAPES[n]))
print("Representative block tensors:")
for n in NAMES:
    if ".layers.0." in n:
        print(n, tuple(SHAPES[n]))
if isinstance(scenes, list):
    print("Suite/level/stage counts:", dict(collections.Counter((s.get("suite"),s.get("level"),s.get("stage")) for s in scenes)))
    discovery, validation = [], []
    for s in scenes:
        h = int(hashlib.sha256(("recovery_diagnosis_v1:" + str(s["id"])).encode()).hexdigest(), 16)
        (discovery if h % 3 == 0 else validation).append(s["id"])
    (W / "scene_split.json").write_text(json.dumps({"discovery": discovery, "validation": validation}, indent=2))
    print("Reserved split:", len(discovery), "discovery,", len(validation), "validation scenes")
(W / "notes.md").write_text(
    "# Recovery weight-edit investigation\n\n"
    "## Protocol\n"
    "- No training or kernel forward passes. Only permitted practice scenes.\n"
    "- Scene-level discovery/validation split saved in scene_split.json before outcomes.\n"
    "- Diagnose with actions, frames, attention, and activation/weight analysis.\n"
    "- Require causal targeted ablations, dose-response, and paired validation.\n"
    "- Ordinary-task evaluation is not explicitly exposed by the tool; check its availability, "
    "and do not substitute recovery preservation for ordinary-task evidence.\n\n"
    "## Pass budget\n- Forward passes used: 0/50.\n\n"
    "## Current edits\n- None.\n"
)
```
