The saved files contain more than episode averages, so I will check whether token-level residuals are available. Those are needed to distinguish spatial variation from image-wide features that could dominate attention.
```python
rr4=json.loads((W/"probe_4"/"results.json").read_text())
z4=np.load(rr4[0]["activations"])
print("Non-mean/call keys:",[(k,z4[k].shape) for k in z4.files if not k.startswith(("mean::","ncalls::","calls::"))][:65])
print("Last keys:",[(k,z4[k].shape) for k in z4.files[-12:]])
print("First catalog item:",json.dumps(cat[0])[:1500])
print("Roll record:",(W/"roll_39.jsonl").read_text().splitlines()[0][:1800])
print("Probe4 inputs:")
for r in rr4:
    print(r["idx"],str(Path(r["input"]).parent.name)[-105:],Path(r["input"]).name)
print("Physical manifest structure:",str(json.loads((W/"physical_grip_probe_manifest.json").read_text()))[:1600])
```
