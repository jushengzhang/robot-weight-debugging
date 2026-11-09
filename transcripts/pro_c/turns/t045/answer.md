The practice list contains only perturbed episodes, so it does not currently provide a direct original-task control. I will keep object and semantic episodes as retention checks, while treating that as an incomplete substitute for the required original-task evaluation.
```python
print("PROBE14 META",[(p.name,p.stat().st_size) for p in (W/"probe_14").iterdir() if p.suffix not in [".npy",".npz"]])
az=np.load(W/"probe_14"/"probe_orig_0_acts.npz")
print("ACT KEYS/SHAPES", [(k,az[k].shape) for k in az.files][:65])
print("ALL DEV IDS", [r["id"] for r in dev_eps])
for p in (W/"probe_14").glob("*.json"):
    print(p.name,p.read_text()[:3500])
print("DIAG FIRST", (W/"diagnostic_episodes.json").read_text()[:1000])
```
