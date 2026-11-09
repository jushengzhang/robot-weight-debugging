I will verify that the frozen edit and final validation panel survived the restart, then use the last pass without further tuning. Afterward, I will apply the precommitted acceptance rule; an edit that does not show a reliable gain will be reverted.
```python
from pathlib import Path
import json
import numpy as np
import torch

w = Path(WORK)
current_edits = list(edited())
print("CURRENT EDITS:", current_edits)
for name in current_edits:
    original = load_original(name)
    delta = load(name) - original
    print(json.dumps({
        "name": name,
        "shape": list(delta.shape),
        "changed_entries": int(torch.count_nonzero(delta)),
        "relative_delta_norm": float(delta.norm() / original.norm()),
        "max_abs_delta": float(delta.abs().max()),
    }))

other_files = sorted(
    p.name for p in w.iterdir()
    if not p.name.startswith(("roll_", "probe_"))
)
print("\nWORK FILES OTHER THAN ROLLOUTS/PROBES:")
print("\n".join(other_files)[:4300])

notes_path = w / "notes.md"
notes_text = notes_path.read_text()
print("\nNOTES BEGINNING:")
print(notes_text[:3800])
print("\nRECENT ROLLOUT FILES:")
for k in (46, 48, 49):
    for p in sorted(w.glob(f"roll_{k}*")):
        print(p.name, "directory" if p.is_dir() else f"{p.stat().st_size} bytes")
```
