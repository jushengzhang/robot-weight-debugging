The first validation batch gained three successes, mostly in environment episodes, but lost one semantic episode. I will continue the precommitted validation unchanged and record the remaining batches so no outcome-based resampling occurs.
```python
import hashlib
for i in range(1,5):
    print("BATCH",i,"IDS",[r["id"] for r in val_batches[i]])
print("frozen edit hashes",[(n,hashlib.sha256(load(n).numpy().tobytes()).hexdigest()[:16]) for n in edited()])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass26: frozen D0.5 validation batch0\n73->76/128,+2.3pp SE2.1; env+3, task+1, semantic-1, other types0. Continue precommitted640; do not tune from this batch.\n")
```
