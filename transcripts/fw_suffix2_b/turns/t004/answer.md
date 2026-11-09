The previous candidate was not convincingly better and lost about three points on paraphrases, so I will not apply it as a default. I’ll create fresh, disjoint diagnostic and validation sets, and inspect the saved detector weights as evidence rather than treating them as a validated repair.```python
import hashlib
from collections import Counter, defaultdict
torch.set_num_threads(min(16, os.cpu_count() or 16))
print("WORK:", str(W))
print("WORK entries:", [p.name for p in W.iterdir()])
pv = json.loads((Path(BENCH) / "practice_variants.json").read_text())
print("Practice structure:", type(pv).__name__)
if isinstance(pv, dict):
    print("Keys:", list(pv)[:12])
    print("Preview:", str(pv)[:1200])
else:
    print("Count:", len(pv), "examples:", pv[:3])
print("Non-block names:")
for n in NAMES:
    if ".blocks." not in n:
        print(n, SHAPES[n])
pc = W / "prior_candidate"
pcfiles = list(pc.glob("*.npy"))
print("Prior candidate files:", len(pcfiles))
for f in pcfiles:
    if "text_embedding" in f.name:
        t = torch.from_numpy(np.load(f))
        nm = f.name[:-4]
        d = t - load_original(nm)
        print(nm, "shape", tuple(t.shape), "changed elements", int((d != 0).sum()),
              "delta_norm", round(float(d.norm()), 4))
notes = """# FastWAM tag repair, new session
## Constraints and protocol
Weight-only analytic edits; no training or local network execution. Read only practice variant list.
Original vs edited paired rollouts; fresh development and reserved validation variants.
## Prior work
Read prior_notes.md. Prior suffix attention gate was weak (+~1.8pp, SE~1.4) and lost ~3pp on paraphrases.
Prior mean-context projectors and head ablations did not restore actions. Do not assume suffix gating sufficient.
## Current state
No edits. 0/50 forward passes used.
## Plan
Fresh matched raw/strip diagnostics; recover richer token-level evidence if needed.
Distinguish direct suffix attention from contextualized-prefix contamination and video-mediated propagation.
Reserve independent raw, clean and language validation; keep only a statistically supported safe edit.
"""
(W / "notes.md").write_text(notes)
```
