I will first inspect the practice-variant list and tensor inventory, then choose matched tasks for diagnosis. I will keep a running log of the evidence, candidate edits, and paired evaluations so that the final choice is based on measured behavior rather than weight statistics alone.
```python
import os, json, math, time, re, shutil
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import sentencepiece as spm

torch.set_grad_enabled(False)
W = Path(WORK)
B = Path(BENCH)
variants_raw = json.loads((B / "practice_variants.json").read_text())
print("WORK:", str(W))
print("variant container:", type(variants_raw).__name__)
if isinstance(variants_raw, dict):
    print("keys:", list(variants_raw)[:12])
    variants = variants_raw.get("variants", variants_raw.get("practice_variants", variants_raw))
else:
    variants = variants_raw
print("variants:", len(variants))
print("examples:", str(variants[:3])[:2200] if isinstance(variants, list) else str(variants)[:2200])
print("tensor count:", len(NAMES))
print("non-block tensors:", [(n, SHAPES[n]) for n in NAMES if not re.search(r"(layers|layer)\.\d+\.", n)][:45])
sp = spm.SentencePieceProcessor(model_file=str(B / "paligemma_tokenizer.model"))
(W / "notes.md").write_text(
    "# Language robustness weight-edit study\n\n"
    "Constraints: no training, no local model execution; practice variants only.\n"
    "Forward passes used: 0/50.\n"
    "Plan: matched-task diagnosis, targeted causal edits with dose response, "
    "paired confirmation on disjoint practice variants, non-language safety checks.\n"
    "Current best: original weights (no edits).\n"
)
```
I will inspect the practice variants and tensor inventory first, then choose matched tasks for diagnosis. I will also keep a running log of the evidence, candidate edits, and evaluations.

```python
import os, json, math, time, re, shutil
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import sentencepiece as spm

torch.set_grad_enabled(False)
W = Path(WORK)
B = Path(BENCH)
variants_raw = json.loads((B / "practice_variants.json").read_text())
print("WORK:", str(W))
print("variant container:", type(variants_raw).__name__)
if isinstance(variants_raw, dict):
    print("keys:", list(variants_raw)[:12])
    variants = variants_raw.get("variants", variants_raw.get("practice_variants", variants_raw))
else:
    variants = variants_raw
print("variants:", len(variants))
print("examples:", str(variants[:3])[:2200] if isinstance(variants, list) else str(variants)[:2200])
print("tensor count:", len(NAMES))
print("non-block tensors:", [(n, SHAPES[n]) for n in NAMES if not re.search(r"(layers|layer)\.\d+\.", n)][:45])
sp = spm.SentencePieceProcessor(model_file=str(B / "paligemma_tokenizer.model"))
(W / "notes.md").write_text(
    "# Language robustness weight-edit study\n\n"
    "Constraints: no training, no local model execution; practice variants only.\n"
    "Forward passes used: 0/50.\n"
    "Plan: matched-task diagnosis, targeted causal edits with dose response, "
    "paired confirmation on disjoint practice variants, non-language safety checks.\n"
    "Current best: original weights (no edits).\n"
)
```
