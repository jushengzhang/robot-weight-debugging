I will quantify this intervention's effect on all seven action dimensions and inspect layer-16 attention, which has not yet been mapped. The result will determine whether to evaluate this self-reinforcement hypothesis in simulation or move the intervention earlier.

```python
pr9=json.loads((W/"probe_9/results.json").read_text())
s9=paired_probe_summary(pr9,{16:[0,1,2,3,4,5,6,7],17:[0,3,4,6]})
print("compact layer16 attention; base/wrist/BOS/other-language/self")
compact_maps([{**p,"attention":{"16":p["attention"]["16"]}} for p in pr9 if p["arm"]=="orig"])
with open(W/"notes.md","a") as f:
    f.write("\nImportant confound: initial open/closed episodes use different saved Gaussian noise (RMS difference1.3751, same noise pair across scenes). Thus early expert open-closed feature contrasts are NOT pure gripper visual contrasts. VLM/vision contrasts are noise-independent. Original/edited paired probes always share exact input/noise.\n")
```
