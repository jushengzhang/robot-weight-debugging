The probe contains 90 activation entries, more than the 30 attention entries inspected so far. I will identify the remaining hook types before choosing an edit, since separate cross-attention outputs would let me target tag injection more directly and avoid using downstream attention differences as a proxy.
```python
print("Probe activation keys:")
for k in probe4_acts[0]:
    print(k, probe4_acts[0][k].shape)
    if len(k)>300:
        break
with np.load(raw_records[0]["activations"]) as a:
    print("Rollout hook types:", sorted({(k.split("::")[0],k.split("::")[1].split(".")[-1])
                                       for k in a.files}))
print("Probe hook types:", sorted({(k.split("::")[0],k.split("::")[1].split(".")[-1])
                                   for k in probe4_acts[0]}))
print("Map rows total attention, raw vs clean:")
for l in ["0","10","12","23"]:
    print(l, [(round(float(np.load(probe3[i]["attention"][l]).sum(-1).mean()),4))
              for i in range(4)])
```
