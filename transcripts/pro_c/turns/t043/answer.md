Before spending a pass, I will verify the sampling settings and the saved activation format. The probe summaries are modality means, so any claim that the edit preserves vision must be limited to those means—not individual image patches.
```python
for p in sorted((W/"fwd"/"roll_6").rglob("*")):
    print(str(p.relative_to(W)), p.stat().st_size)
    if p.is_file() and p.stat().st_size < 18000:
        print(p.read_text()[:18000])
print("DEV SEEDS", Counter(r["seed"] for r in rows6 if r["model"]=="original"))
practice = json.loads((pathlib.Path(BENCH)/"practice_episodes.json").read_text())
print("PRACTICE",len(practice),Counter(r["ptype"] for r in practice),practice[:2])
ix14=json.loads((W/"probe_14"/"index.json").read_text())
print("PROBE14 INDEX TYPE",type(ix14))
print(str(ix14)[:4500])
az=np.load(W/"probe_14"/"probe_orig_0_acts.npz")
print("ACT KEYS/SHAPES", [(k,az[k].shape) for k in az.files][:50])
print("ALL DEV IDS",[r["id"] for r in dev_eps])
```
