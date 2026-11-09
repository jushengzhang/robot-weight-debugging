I’ll recover the saved candidate, validation panels, and diagnostic artifacts, then record the latest result. I’ll keep the frozen edit unchanged while finishing its independent camera validation, so these results do not become another tuning set.
```python
from pathlib import Path
import json, hashlib, numpy as np, torch

root = Path(WORK)
torch.set_num_threads(16)
print("WORK:", str(root))
print("Active edits:", edited())
print("Top-level artifacts:")
for p in sorted(root.iterdir()):
    if not p.name.startswith(("roll_", "probe_")):
        print(p.name, "dir" if p.is_dir() else f"{p.stat().st_size} bytes")
lock_path = root / "candidate_lock.json"
if lock_path.exists():
    print("Candidate lock:", json.dumps(json.loads(lock_path.read_text()), indent=2)[:5000])
print("Recent notes:")
print((root / "notes.md").read_text()[-4500:])
with (root / "notes.md").open("a") as f:
    f.write(
        "\n## Pass21: second independent camera panel\n"
        "- Validation B:55/128 original vs59/128 edited; +3.125pp SE3.3 (11 gains/7 losses).\n"
        "- Combined A+B:114/256 vs116/256, +0.781pp; discovery gain not yet replicated.\n"
        "- Candidate remains frozen; finish preplanned C/D before interpreting. Passes21/40.\n"
    )
```
