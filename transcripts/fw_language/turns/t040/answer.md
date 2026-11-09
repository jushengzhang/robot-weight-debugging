The probe uses different camera labels from the rollout summaries. I will identify those labels and rerun the comparison without assuming that the two artifact schemas are identical.
```python
print("Probe video keys:",[(k,v.shape) for k,v in p4acts[0].items() if "video" in k][:12])
print("Probe non-action prefixes:",sorted({k.split("::")[0] for k in p4acts[0] if "action." not in k}))
print("Last keys:",list(p4acts[0])[-8:])
```
