The gate improves the Camera action gap by about 18% but worsens Initial States, leaving the overall gap 15% worse. That is not a basis for keeping it. I’ll run the reserved 96-episode paired development set to see whether these action changes help or hurt task success, then use that result to decide whether further gate tests are warranted.
```python
splits=json.loads((W/"splits.json").read_text())
dev=[{k:e[k] for k in ("suite","id","seed")} for e in splits["development"]]
print("DEVELOPMENT REQUEST:")
print(json.dumps({"type":"rollout","model":"both","episodes":dev},separators=(",",":")))
base=np.stack([np.load(W/"probe_3"/f"probe_orig_{i+24}_action.npy") for i in range(24)])
e20=np.stack([np.load(W/"probe_20"/f"probe_edit_{i+24}_action.npy") for i in range(24)])
cd=np.sqrt(((e20-base)**2).mean((1,2)))
print("Largest clean drifts:",[(int(i),items[i]["category"],round(float(cd[i]),5)) for i in np.argsort(cd)[-5:]])
with (W/"notes.md").open("a") as f:
    f.write("\nPass20 unmatched gate: gap ratio1.151 overall, Camera.816, Initial1.552, clean drift.00698, language.00144. "
            "Probe is mixed, not a retention basis. Paired independent development96 rollout planned before further gate tuning.\n")
```
