The probe exposes a 7-by-14 video-token grid, with seven columns per camera. I will now rank layers by fixed-input texture sensitivity and inspect the recorded tensors for head-resolved information. This ranking will guide a small number of attention-map probes rather than a blind sweep of weight edits.

```python
pr3=json.loads((W/"probe_3"/"results.json").read_text())
print("Probe results type:",type(pr3).__name__)
if isinstance(pr3,dict):
    print("Keys:",list(pr3))
    print("Preview:",str(pr3)[:2000])
else:
    print("First result:",pr3[0])
```
