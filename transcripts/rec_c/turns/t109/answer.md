I will check whether the intervention actually produces upward motion preferentially in empty-closure states. The feature constraints were built from token-averaged activations, whereas attention reads individual tokens, so the probe is essential for detecting leakage into contact and open-gripper states.
```python
p37=json.loads((W/"probe_37"/"results.json").read_text())
b37={(r["idx"],r["arm"]):r for r in p37}
summ37=collections.defaultdict(list);effects37=[]
upstream_vlm=[];upstream_expert=[]
for i,m in enumerate(value_manifest):
    oo=np.load(b37[i,"orig"]["action"]);ee=np.load(b37[i,"edit"]["action"])
    dd=ee[:5]-oo[:5]
    vals={"dz":float(dd[:,2].mean()),"xy_rms":float(np.sqrt(np.mean(dd[:,:2]**2))),
          "rot_rms":float(np.sqrt(np.mean(dd[:,3:6]**2))),
          "grip_sign_change":float(np.mean((ee[:5,6]>0)!=(oo[:5,6]>0))),
          "xyz_rms":float(np.sqrt(np.mean(dd[:,:3]**2)))}
    label=m["split"]+"_"+m["role"]
    summ37[label].append(vals)
    effects37.append(vals)
    oz=np.load(b37[i,"orig"]["activations"]); ez=np.load(b37[i,"edit"]["activations"])
    upstream_vlm.append(np.max(np.abs(oz["mean::vlm.16.resid::wrist"]-ez["mean::vlm.16.resid::wrist"])))
    upstream_expert.append(np.max(np.abs(oz["calls::expert.16.resid::action"][0]-ez["calls::expert.16.resid::action"][0])))
for label,rs in summ37.items():
    print(label,len(rs),{k:round(float(np.mean([r[k] for r in rs])),4) for k in rs[0]},
          "dz min/max",round(min(r["dz"] for r in rs),3),round(max(r["dz"] for r in rs),3))
scores=(X35["wrist"]@d).numpy()
dz=np.array([v["dz"] for v in effects37])
print("feature-score vs actual dz correlation",round(float(np.corrcoef(scores,dz)[0,1]),3))
print("upstream maximum difference VLM16/expert16 firstcall",max(upstream_vlm),max(upstream_expert))
print("empty upward counts",sum(effects37[i]["dz"]>0 for i in range(0,44,2)),"/22")
with (W/"notes.md").open("a") as f:
    f.write("\nPass37 actual selective-clearance probe results:\n")
    for label,rs in summ37.items():
        f.write(label+" "+str({k:round(float(np.mean([r[k] for r in rs])),4) for k in rs[0]})+"\n")
    f.write("Upstream VLM16 and first-call expert16 unchanged exactly. "
            f"Feature/dz correlation={np.corrcoef(scores,dz)[0,1]:.3f}. "
            "Token-mean protection is not protection of individual attention-weighted patches.\n")
```
