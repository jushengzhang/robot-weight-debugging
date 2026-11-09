I’ll reload the frozen confirmation plan and completed results, then run the remaining two batches without changing the candidate or selection rule. So far, the confirmation data show no demonstrated gain.
```python
from pathlib import Path
import json
import numpy as np

work = Path(WORK)
plan_path = work / "final_confirmation_plan.json"
plan = json.loads(plan_path.read_text())

print("Plan keys:", list(plan) if isinstance(plan, dict) else type(plan).__name__)
if isinstance(plan, dict):
    for k, v in plan.items():
        if isinstance(v, list):
            print(k, "list length", len(v))
            if v:
                item = v[0]
                print(" first item:", str(item)[:600])
        elif isinstance(v, dict):
            print(k, "dict keys:", list(v)[:20])
        else:
            print(k, str(v)[:700])
print("Current edited tensors:", list(edited()))
print("Recent rollout files:",
      [p.name for p in sorted(work.glob("roll_*.jsonl"),
                             key=lambda p: int(p.stem.split("_")[1]))[-8:]])
print("Other confirmation files:",
      [p.name for p in work.iterdir() if "confirm" in p.name.lower()])
```
