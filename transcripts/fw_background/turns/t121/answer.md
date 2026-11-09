The latest MLP edit reduced background success by 4.6 points and also lost a light-condition episode, so I will remove it. I will audit the saved evidence before choosing how to use the remaining ten passes; none of the edits tested so far has earned acceptance.
```python
from pathlib import Path
import json, numpy as np, torch, math, re, collections

W = Path(WORK)
torch.set_num_threads(12)

removed = list(edited())
for name in removed:
    drop_edit(name)

with (W / "notes.md").open("a") as f:
    f.write(
        "\n## Pass30: early MLP candidate rejected\n"
        "- Paired diagnostic: BG50/65 original vs47/65 edited; light36/37 vs35/37. "
        "Overall1 rescue,5 regressions; -3.9pp(SE2.4pp). Fixed-input action changes were near replay noise.\n"
        "- Removed all current edits. No candidate accepted. Ten forward passes remain; "
        "independent validation has not yet been used.\n"
    )

print("Removed edits:", removed)
print("Current edited tensors:", list(edited()))
print("WORK files:")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    else:
        print(p.name + "/")
print("\nInput/output and normalization tensor names:")
for name in NAMES:
    if any(s in name for s in ("patch_embedding", "head.", "norm_q", "norm_k")):
        if ".blocks." not in name or ".blocks.0." in name:
            print(name, SHAPES[name])
```
