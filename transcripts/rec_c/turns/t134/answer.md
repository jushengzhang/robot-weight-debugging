I will summarize the probe and, if the intended selectivity remains, freeze this edit for the untouched confirmation batch. Rollout success—not the feature separation alone—will determine whether it is worth retaining.
```python
a44 = probe_actions(44)
d44 = a44["edit"]-a44["orig"]
res44 = {}
for group in dict.fromkeys(group_key(r) for r in m):
    inds = [i for i,r in enumerate(m) if group_key(r)==group]
    d = d44[inds,:5]
    res44[group] = {
        "n":len(inds),
        "dz_mean":round(float(d[:,:,2].mean()),4),
        "dz_RMS":round(float(np.sqrt(np.mean(d[:,:,2]**2))),4),
        "xy_RMS":round(float(np.sqrt(np.mean(d[:,:,:2]**2))),4),
        "rot_RMS":round(float(np.sqrt(np.mean(d[:,:,3:6]**2))),4),
        "grip_sign_change":round(float(np.mean((a44["orig"][inds,:5,6]>0)!=(a44["edit"][inds,:5,6]>0))),4)
    }
    print(group, res44[group])
confirm3 = json.loads((W/"confirmation_batch3.json").read_text())
print("Untouched confirmation scenes:", len(confirm3), confirm3)
# Confirm no confirmation scene has appeared in any saved rollout record.
seen = set()
for p in W.glob("roll_*.jsonl"):
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "id" in r:
                seen.add(r["id"])
print("Overlap with rolled IDs:", sorted(set(confirm3)&seen))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass44: H1-only centered clearance0.60 probe\n")
    for k,v in res44.items():
        f.write(k+" "+json.dumps(v)+"\n")
```
