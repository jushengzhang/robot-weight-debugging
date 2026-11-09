I will first recover the saved analysis and confirm the current edit. With 13 forward passes left, I will test the spatial-attention hypothesis and reserve enough passes for an independent paired evaluation if it shows promise.
```python
from pathlib import Path
import json, numpy as np, torch, math, os
W = Path(WORK)
torch.set_num_threads(16)
print("Current edits:", edited())
print("Work files:")
for p in sorted(W.iterdir()):
    if not p.name.startswith(("roll_", "probe_")):
        print(p.name, (f"{p.stat().st_size} bytes" if p.is_file() else "directory"))
print("\nRecent notes:")
print((W / "notes.md").read_text()[-6500:])
qname = "paligemma_with_expert.gemma_expert.model.layers.8.self_attn.q_proj.weight"
q0, q1 = load_original(qname), load(qname)
print("\nPer-head relative query changes:",
      [round(float((q1[h*256:(h+1)*256]-q0[h*256:(h+1)*256]).norm()
                   / q0[h*256:(h+1)*256].norm()), 4) for h in range(8)])
```
