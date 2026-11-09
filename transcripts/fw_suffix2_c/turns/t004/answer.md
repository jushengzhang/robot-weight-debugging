The prior notes also show a paraphrase loss, so I will not treat that candidate as a safe starting repair. I will establish a fresh matched diagnostic set and reserve separate variants for validation; the first comparison will change only whether the instruction includes the tag.
```python
import hashlib
torch.set_num_threads(min(24, os.cpu_count() or 24))
pvdata = json.loads((Path(BENCH) / "practice_variants.json").read_text())
print("JSON TYPE:", type(pvdata).__name__)
if isinstance(pvdata, dict):
    print("KEYS:", list(pvdata)[:20])
    variants = pvdata.get("variants", pvdata.get("practice_variants", []))
else:
    variants = pvdata
print("VARIANTS:", len(variants))
print("EXAMPLES:", variants[:3])
from collections import Counter, defaultdict
print("CATEGORIES:", Counter(v["category"] for v in variants))
print("SUITES:", Counter(v["suite"] for v in variants))
print("VIDEO BLOCK0 SHAPES:", [(n.split(".blocks.0.")[-1], SHAPES[n]) for n in NAMES if ".video.blocks.0." in n])
print("PRIOR CANDIDATE:", len(list((W / "prior_candidate").glob("*.npy"))))
(W / "notes.md").write_text(
    "# FastWAM second session\n\n"
    "Read prior_notes.md. Earlier linear/head edits failed; selective context gate gave small tagged gain and language loss. "
    "No prior edits applied. Goal: diagnose token-level contamination and separate video/action routes; reserve independent validation.\n"
    "Constraints: weight edits only, no local network execution or training. 0/50 forward passes used.\n"
)
```
