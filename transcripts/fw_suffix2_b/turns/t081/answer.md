I’ll first recover the saved edit and notes, including the missing Language Instructions result, so we can judge its safety before using the remaining passes for validation.
```python
from pathlib import Path
import json, numpy as np, torch
W = Path(WORK)
torch.set_num_threads(12)

print("PRIOR NOTES")
print((W / "prior_notes.md").read_text()[:15000])
print("\nWORK FILES")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    elif not p.name.startswith(("roll_", "probe_")):
        print(p.name + "/")
print("\nCURRENT EDITS:", len(edited()))
print(list(edited())[:12])

for k in (28, 29):
    p = W / f"roll_{k}.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        print("\nROLLOUT", k, "rows", len(rows))
        print("first record:", json.dumps(rows[0])[:1800])
        groups = {}
        for r in rows:
            m = str(r.get("model", "unknown"))
            g = groups.setdefault(m, [0, 0])
            g[0] += int(bool(r.get("success", False)))
            g[1] += 1
        print("success counts:", groups)
```
