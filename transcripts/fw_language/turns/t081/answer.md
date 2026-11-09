I will restore the frozen validation plan and verify that only the intended 128 query-scale entries are edited. The candidate remains provisional; I will judge it on untouched variants without tuning it to their results.
```python
from pathlib import Path
import json, numpy as np, torch

work = Path(WORK)
plan = json.loads((work / "locked_language_validation_plan.json").read_text())
print("Plan structure:", list(plan) if isinstance(plan, dict) else f"list[{len(plan)}]")
if isinstance(plan, dict):
    for k, v in plan.items():
        if isinstance(v, list):
            print(k, "length", len(v), "first-item keys",
                  list(v[0]) if v and isinstance(v[0], dict) else None)
        elif not isinstance(v, dict):
            print(k, str(v)[:250])

active = list(edited())
print("Active edits:", active)
name = "mot.mixtures.action.blocks.11.self_attn.norm_q.weight"
assert active == [name] or set(active) == {name}
original, current = load_original(name), load(name)
changed = torch.nonzero(current.reshape(-1) != original.reshape(-1)).flatten()
expected = original.clone()
expected.reshape(-1)[19*128:20*128] *= 0.75
assert torch.equal(current, expected)
print("Verified edit:", {
    "changed_values": changed.numel(),
    "first_index": int(changed.min()),
    "last_index": int(changed.max()),
    "relative_delta": float(torch.linalg.vector_norm(current-original) /
                            torch.linalg.vector_norm(original))
})
print("Relevant saved results:",
      sorted(p.name for p in work.glob("*") if
             p.name.startswith(("roll_", "probe16", "candidate_L11", "locked_language")))
```
