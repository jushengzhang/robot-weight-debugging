The saved rollout folder contains binary action files, not a request log. I will use the episode records to reconstruct the sample and inspect only JSON metadata and array shapes.
```python
print("DEV SEEDS", Counter(r["seed"] for r in rows6 if r["model"]=="original"))
practice = json.loads((pathlib.Path(BENCH)/"practice_episodes.json").read_text())
print("PRACTICE", len(practice), Counter(r["ptype"] for r in practice), practice[:2])
ix14=json.loads((W/"probe_14"/"index.json").read_text())
print("PROBE14 INDEX",str(ix14)[:5000])
az=np.load(W/"probe_14"/"probe_orig_0_acts.npz")
print("ACT KEYS/SHAPES", [(k,az[k].shape) for k in az.files][:65])
print("ALL DEV IDS", [r["id"] for r in dev_eps])
```
