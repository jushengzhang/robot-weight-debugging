I will locate the saved input paths and attention-map layout before making the probe request. This avoids spending a pass on observations that cannot be matched to the original comparison.
```python
p=W/"probe_36"
print("Probe 36 metadata files:", [(x.name,x.stat().st_size) for x in p.iterdir() if x.suffix not in [".npy",".npz"]])
for x in p.iterdir():
    if x.suffix==".json":
        j=json.loads(x.read_text())
        print(x.name, str(j)[:5000])
print("fwd probe entries:", [x.name for x in (W/"fwd").iterdir() if "36" in x.name or "2"==x.name])
for x in (W/"fwd").rglob("*36*"):
    print(x)
print("Diagnosis selection:", (W/"diagnosis_episodes.json").read_text()[:2200])
cat=json.loads((W/"observation_catalog.json").read_text())
print("Catalog:",type(cat), str(cat)[:1800])
```
