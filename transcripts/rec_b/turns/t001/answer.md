I will first inspect the practice-scene metadata and tensor layout, then reserve separate scenes for diagnosis and validation. I will keep a running record of hypotheses, causal tests, and paired results so the final edit is supported by evidence rather than a favorable small sample.
```python
import os, json, math, collections, pathlib, numpy as np, torch, time
torch.set_grad_enabled(False)
W = pathlib.Path(WORK)
B = pathlib.Path(BENCH)
scenes_raw = json.loads((B/"practice_scenes.json").read_text())
print("practice format:", type(scenes_raw).__name__)
if isinstance(scenes_raw, dict):
    print("keys:", list(scenes_raw)[:20])
    scenes = scenes_raw.get("scenes", scenes_raw.get("practice_scenes", scenes_raw))
else:
    scenes = scenes_raw
print("number:", len(scenes))
print("examples:", json.dumps(scenes[:3] if isinstance(scenes,list) else list(scenes.items())[:3], ensure_ascii=True)[:3500])
if isinstance(scenes,list):
    for k in ["suite","level","stage"]:
        print(k, dict(collections.Counter(s.get(k) for s in scenes)))
print("tensors:", len(NAMES))
for n in NAMES:
    if ("layers.0." in n or not any(x in n for x in ["layers.", "encoder.layers."])) and ("vision_model" not in n):
        print(n, SHAPES[n] if isinstance(SHAPES,dict) else SHAPES[NAMES.index(n)])
(W/"notes.md").write_text(
    "# Recovery weight-edit investigation\n"
    "## Plan\n"
    "- Inspect baseline trajectories, visual evidence, and internal activations before editing.\n"
    "- Split practice scenes into diagnostic/tuning and untouched validation partitions.\n"
    "- Test mechanism-specific, nonuniform weight edits with paired comparisons and dose-response.\n"
    "- Check preservation on baseline-solved scenes; ordinary LIBERO testing only if supported by the tool.\n"
    "- Budget: 50 forward requests; no training or local model execution.\n"
    "## Current edit\nNone.\n")
```
