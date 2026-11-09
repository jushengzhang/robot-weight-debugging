The weight-installation block timed out and restarted the kernel, so I will verify the on-disk state before any probe. I will rebuild the half-dose candidate from originals if necessary; a partially installed edit must not be evaluated.
```python
from pathlib import Path
import json, numpy as np, torch, shutil, os, time
W=Path(WORK)
torch.set_num_threads(12)
torch.set_grad_enabled(False)
print("CURRENT", (W/"current_candidate.json").read_text())
ens=list(edited())
print("EDIT COUNT",len(ens),"FIRST",ens[:5])
print("CANDIDATES", [p.name for p in (W/"candidates").iterdir()])
print("NOTES TAIL", (W/"notes.md").read_text()[-1800:])
print("EDIT FILE COUNTS",len(list((W/"edited").glob("*.npy"))))
print("TEXT FILES")
for n in ens:
    if "text_embedding" in n:
        p=W/"edited"/(n+".npy")
        print(n, p.stat().st_size if p.exists() else "no-file")
```
