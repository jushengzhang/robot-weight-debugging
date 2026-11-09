I’ll inspect the probe tensors and look for directions that separate real instruction tokens from BOS and image tokens. Any edit will act only on a small identified subspace, rather than rescaling an entire layer or changing action/noise scales.

```python
pr3 = json.loads((W/"probe_3"/"results.json").read_text())
print("probe record:", str(pr3[0])[:1800])
p3a = np.load(workpath(pr3[0]["activations"]))
print("arrays:", [(k,p3a[k].shape) for k in p3a.files])
# Inspect near-deterministic visual attention for sinks rather than object tracking.
for l in (3,6,9,12):
    maps=[np.load(workpath(r["attention"][str(l)])).mean(1) for r in pr2]
    print("L",l, "top image token per head, four inputs:",
          [[int(np.argmax(m[h,:512])) for m in maps] for h in range(8)])
    print("top-image probability (four-input mean):",
          np.round(np.mean([[m[h,:512].max() for h in range(8)] for m in maps],0),3).tolist())
print("WORK size of probe3 MB:",
      round(sum(p.stat().st_size for p in (W/"probe_3").rglob("*") if p.is_file())/1e6,1))
```
