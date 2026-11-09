The probe returned all 57 observations and confirmed the video-token geometry: a 7x14 grid, with seven columns per camera. I will now locate the earliest and largest same-state camera shifts, compare action discrepancies, and select a small set of layers and episodes for head-level attention maps.```python
log_pass(2,"probe","All-layer first-state camera/control comparison",extra={"inputs":57,"noise_seed":46001})
p2raw=json.loads((W/"probe_2"/"results.json").read_text())
print("Probe result type:",type(p2raw).__name__)
if isinstance(p2raw,dict):
    print("Keys:",list(p2raw))
    print(str(p2raw)[:2000])
else:
    print("First record:",json.dumps(p2raw[0],indent=2)[:2200])
```
