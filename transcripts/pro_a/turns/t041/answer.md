The row-193-only edit improved task following on the development set, but its overall gain is still uncertain and it did not help position perturbations. I will recover the saved edits, episode partitions, and diagnostics before deciding which mechanism to test next; the independent validation episodes will remain separate.
I will also record pass 15 in the notes so that the current edit and its measured trade-offs are preserved across restarts.
```python
from pathlib import Path
import json, numpy as np, torch, hashlib
W = Path(WORK)
print("WORK:", str(W))
print("Files:", sorted(p.name for p in W.iterdir()))
print("Current edits:", edited())

notes = W / "notes.md"
with notes.open("a") as f:
    f.write(
        "\n## Pass 15: row193 only, positional residual retain20%\n"
        "Dev12869->73, +3.1 +/-2.5pp SE, 7 wins/3 losses. "
        "Task5->8, position3 unchanged, environment17 unchanged, "
        "object23->22, semantic21->23; old-goal task failures13->9. "
        "Still provisional; no independent validation yet.\n"
    )

episodes = json.loads((Path(BENCH) / "practice_episodes.json").read_text())
print("Episode container:", type(episodes).__name__)
if isinstance(episodes, dict):
    print("Container keys:", list(episodes)[:15])
else:
    print("Practice count:", len(episodes), "first:", episodes[0])

for fname in ["roll_1.jsonl", "roll_8.jsonl", "roll_15.jsonl"]:
    p = W / fname
    if p.exists():
        rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        print(fname, "records:", len(rows), "keys:", list(rows[0]))
        print("First record:", {k: v for k, v in rows[0].items()
                                if k not in ["actions", "activations"]})
for dname in ["probe_2", "probe_3", "probe_9", "probe_13"]:
    d = W / dname
    if d.exists():
        print(dname, "files:", [p.name for p in list(d.iterdir())[:12]])
```
