I will measure the gate’s effect and compare changes by category. I will use those results to decide whether to test a stronger dose or abandon this detector family.
```python
m11,act11=probe_metrics(11)
# Keep a compact reproducible checkpoint of this candidate before any dose change.
def snapshot_edit(label):
    d=W/label
    d.mkdir(exist_ok=True)
    for n in edited():
        np.save(d/f"{n}.npy",load(n).numpy())
    return d
if m11["ratio"]<1:
    snapshot_edit("candidate_gate_p11")
print("RAW first10 action-step gap ratio:",
      rms(act11[:24,:10]-base_actions[24:48,:10])/rms(base_actions[:24,:10]-base_actions[24:48,:10]))
print("TRANSLATION-only ratio:",
      rms(act11[:24,:,:3]-base_actions[24:48,:,:3])/rms(base_actions[:24,:,:3]-base_actions[24:48,:,:3]))
```
