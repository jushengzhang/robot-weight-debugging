I will check which saved probes already contain first-observation vision features. Using initial observations avoids confusing the environment shift with the different robot trajectories of successful and failed episodes.
```python
for k in [2,3,4,5,7,10,14,17,19]:
    rr=json.loads((W/f"probe_{k}"/"results.json").read_text())
    entry=next((r for r in rr if "activations" in r),None)
    keys=[]
    if entry:
        zz=np.load(entry["activations"])
        keys=[k for k in zz.files if k.startswith("mean::vision")]
    print(k,"inputs",len(rr),"vision",keys[:10])
# Compare episode-average environment contrasts to see which vision depth is plausible;
# a fresh initial-observation probe will be used if an edit is based on these directions.
by_scene=defaultdict(dict)
for r in r1:by_scene[(r["suite"],r["task"])][r["ptype"]]=r
for l in [0,8,16,26]:
    differences=[]
    refs=[]
    for group in by_scene.values():
        if "environment" not in group or "semantic" not in group:continue
        e=np.load(group["environment"]["activations"])
        s=np.load(group["semantic"]["activations"])
        for cam in ["base","wrist"]:
            key=f"mean::vision.{l}.resid::{cam}"
            differences.append(e[key]-s[key]);refs.append(s[key])
    d=torch.tensor(np.stack(differences))
    ss=torch.linalg.svdvals(d)
    print("episode-average layer",l,"rank1",round(float(ss[0]**2/(ss**2).sum()),3),
          "rank4",round(float((ss[:4]**2).sum()/(ss**2).sum()),3))
```
