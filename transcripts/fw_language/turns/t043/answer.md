The probe uses a different metadata filename than the rollout records. I will locate it and summarize the layerwise contrasts, keeping episode-averaged statistics separate from the controlled, same-observation probes.
```python
print("Probe metadata:", [p.name for p in (W/"probe_4").iterdir() if p.suffix not in [".npy",".npz"]])
for p in (W/"probe_4").iterdir():
    if p.suffix == ".json":
        print(p.name, p.read_text()[:2300])
r3 = [json.loads(s) for s in (W/"roll_3.jsonl").read_text().splitlines()]
pairs = json.loads((W/"natural_probe_pairs.json").read_text())
print("\nFirst record:", compact_obj(r3[0], 2500))
z = np.load(W/"probe_4/probe_orig_0_acts.npz")
print("\nKeys:", len(z.files), z.files[:8], z.files[-4:])
print("Shapes:", [(k,z[k].shape) for k in z.files[:8]])
print("\nRollout index:", (W/"roll_3_acts/index.json").read_text()[:1800])
print("\nPair list:")
for i,p in enumerate(pairs):
    print(i, p["good"]["suite"],p["good"]["id"],p["bad"]["id"],
          "d",p["good"]["difficulty"],p["bad"]["difficulty"],
          "bad prompt:", p["bad"]["instruction"][:150])
print("\nInput files:")
fd = Path(pairs[0]["good"]["frames_dir"])
print([p.name for p in fd.iterdir()][:15])
```
