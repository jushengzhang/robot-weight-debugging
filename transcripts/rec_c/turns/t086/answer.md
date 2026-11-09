Most of the attention-to-BOS behavior is shared across the diagnostic observations, so it is not yet a good failure-specific target. I will instead check whether loop gripper reversals are weak, ambiguous commands or decisive alternations within action chunks. That distinction matters before changing the gripper’s feedback path through the action-input projection.
```python
groups=collections.defaultdict(list)
for r in r1:
    a=np.load(r["actions_npy"])
    if a.ndim!=2 or a.shape[1]<7: continue
    key="success" if r["success"] else ("loop" if r["grip_flips"]>=6 else "other_failure")
    g=a[:,6]; flips=np.flatnonzero((g[1:]>0)!=(g[:-1]>0))+1
    groups[key].append({
        "n":len(a), "gabs":float(np.mean(np.abs(g))),
        "ambiguous":float(np.mean(np.abs(g)<0.25)),
        "flipmag":float(np.mean(np.abs(g[flips]))) if len(flips) else np.nan,
        "z":float(a[:,2].mean()),"zneg":float(np.mean(a[:,2]<-0.05)),
        "xyz":float(np.linalg.norm(a[:,:3],axis=1).mean()),
        "flips_inside":int(np.sum(flips%5!=0)), "flips_total":len(flips)})
for key,rs in groups.items():
    print(key,"episodes",len(rs),
          {k:round(float(np.nanmean([r[k] for r in rs])),3)
           for k in ["gabs","ambiguous","flipmag","z","zneg","xyz"]},
          "inside/all flips",sum(r["flips_inside"] for r in rs),sum(r["flips_total"] for r in rs))
for sid in [979,562,70]:
    for r in r1:
        if r["id"]!=sid: continue
        a=np.load(r["actions_npy"])
        print("scene",sid,r["grip"],"success",r["success"],"flip",r["grip_flips"],
              "grip quantiles",[round(float(x),2) for x in np.quantile(a[:,6],[0,.1,.25,.5,.75,.9,1])],
              "z quartiles",[round(float(x),2) for x in np.quantile(a[:,2],[.1,.5,.9])])
print("saved input keys")
meta3=json.loads((W/"probe_3"/"results.json").read_text())
o=torch.load(meta3[1]["input"],map_location="cpu",weights_only=False)
for key,val in o.items():
    if isinstance(val,dict):
        print(key,[(k,tuple(v.shape) if hasattr(v,"shape") else type(v).__name__) for k,v in val.items()])
    else:
        print(key,tuple(val.shape) if hasattr(val,"shape") else str(val)[:300])
print("last17 action/norm names",[(n,SHAPES[n]) for n in NAMES
    if n.startswith(EXP+".norm.") or n.startswith("time_mlp")])
```
