I will summarize each head's attention by modality and by instruction word, checking masks and array structure first. This will distinguish broad visual attention from attention that actually reaches the requested object name, and provide a basis for a targeted head intervention.
```python
p2 = json.loads((W/"probe_2/results.json").read_text())
print("Probe result structure:", str(p2)[:5500])
# Check per-layer norms directly, avoiding prior aggregate code.
print("Single-episode residual norms:")
for stream,part in [("vlm","base"),("vlm","lang"),("expert","action")]:
    print(stream, part, [(l,round(float(np.linalg.norm(a0[f"mean::{stream}.{l}.resid::{part}"])),3))
                        for l in [0,3,6,9,12,15,17]])
nt = (W/"notes.md").read_text().replace(
    "Rollout resid summaries are identical across requested layers; do not infer layer localization from these.",
    "Rounded aggregate resid norm display was similar across layers, but exact equality checks were FALSE. "
    "Do not infer localization from the aggregate display; use per-input probes.")
(W/"notes.md").write_text(nt)
```
