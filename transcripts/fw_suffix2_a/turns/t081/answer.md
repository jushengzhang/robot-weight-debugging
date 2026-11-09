I’ll read the prior notes and saved helper code first to confirm exactly which edit is active and how the validation sets were reserved. Then I’ll finish the dose test before deciding whether the video-only gate merits larger, independent tests.
```python
from pathlib import Path
import json, numpy as np, torch

W = Path(WORK)
print("=== PRIOR NOTES ===")
print((W / "prior_notes.md").read_text())
print("\n=== WORK FILES ===")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    elif not p.name.startswith(("roll_", "probe_")):
        print(p.name + "/")
print("\n=== ACTIVE EDITS ===")
en = list(edited())
print("count:", len(en))
print("\n".join(en[:10]))
print("\n=== CANDIDATE HELPERS ===")
print((W / "candidate_helpers.py").read_text())
```
