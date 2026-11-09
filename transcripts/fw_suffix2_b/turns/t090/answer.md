The expanded gate matched, but did not improve, the single-detector result and changed more clean-language features. I’ll discard it and freeze the simpler rank-256 hybrid. The remaining passes will be independent paired validation: tagged instructions, matched clean instructions, and paraphrases. I’ll retain it only if the tagged gain is supported and neither safety condition drops materially.
```python
import hashlib, shutil
base = W / "candidates/hybrid_r256_d1_t05_g128"
for ex in ("action","video"):
    for suffix in ("0.weight","0.bias","2.weight","2.bias"):
        n = f"mot.mixtures.{ex}.text_embedding.{suffix}"
        save_edit(n, torch.from_numpy(np.load(base/(n+".npy"))).clone())
cfg = json.loads((base/"candidate.json").read_text())
cfg["frozen_at_pass"] = 31
(W/"current_candidate.json").write_text(json.dumps(cfg,indent=2))

payload = json.loads((Path(BENCH)/"practice_variants.json").read_text())
print("Practice file type:", type(payload).__name__)
if isinstance(payload, list):
    variants = payload
elif isinstance(payload, dict) and "variants" in payload:
    variants = payload["variants"]
else:
    raise ValueError("Inspect practice structure before selecting further variants.")
from collections import Counter
print("Practice category counts:", dict(Counter(v["category"] for v in variants)))
langdev = json.loads((W/"language_development.json").read_text())
print("Language development type/count:", type(langdev).__name__, len(langdev))
print("Frozen:", cfg, "tensor count:", len(edited()))

with (W/"notes.md").open("a") as f:
    f.write("\nPass31 expanded7-detector hybrid raw development:48->52/96, "
            "+4.17pp SE4.17,10gains/6losses. No net improvement over simpler gate; "
            "expanded gate discarded, all8 text embedding tensors restored exactly "
            "from hybrid_r256_d1_t05_g128.\n")
    f.write("\nCANDIDATE FROZEN AFTER PASS31: broad rank256 oblique projection dose1 "
            "plus single suffix detector (threshold.5,gain128,strength6), both experts, "
            "308 tensors. No further tuning. Plan remaining19 passes: "
            "7x128 raw,6x128 strip matched to first6 raw batches,6x128 language. "
            "Use reserved splits first, then new disjoint practice variants where available. "
            "Retain only with independent tagged gain clearly exceeding its SE and "
            "clean/paraphrase point losses no worse than approximately2pp; otherwise revert.\n")
```
