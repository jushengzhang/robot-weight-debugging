I will compare the two doses numerically. A roughly fourfold vertical response with little change in the other readouts would strengthen the causal localization, but the rollout—not that linearity—will determine whether the edit is useful.
```python
p39=json.loads((W/"probe_39"/"results.json").read_text())
b39={(r["idx"],r["arm"]):r for r in p39}
summ39=collections.defaultdict(list);effects39=[]
for i,m in enumerate(value_manifest):
    oo=np.load(b39[i,"orig"]["action"]);ee=np.load(b39[i,"edit"]["action"])
    dd=ee[:5]-oo[:5]
    vals={"dz":float(dd[:,2].mean()),"xy_rms":float(np.sqrt(np.mean(dd[:,:2]**2))),
          "rot_rms":float(np.sqrt(np.mean(dd[:,3:6]**2))),
          "grip_sign_change":float(np.mean((ee[:5,6]>0)!=(oo[:5,6]>0))),
          "xyz_rms":float(np.sqrt(np.mean(dd[:,:3]**2)))}
    summ39[m["split"]+"_"+m["role"]].append(vals);effects39.append(vals)
for label,rs in summ39.items():
    print(label,{k:round(float(np.mean([r[k] for r in rs])),4) for k in rs[0]})
zlo=np.array([r["dz"] for r in effects37]); zhi=np.array([r["dz"] for r in effects39])
print("dz dose slope through zero",round(float((zlo@zhi)/(zlo@zlo)),3),
      "dose correlation",round(float(np.corrcoef(zlo,zhi)[0,1]),3),
      "max high-dose dz",round(float(zhi.max()),3))
with (W/"notes.md").open("a") as f:
    f.write("\nPass39 clearance0.60 probe:\n")
    for label,rs in summ39.items():
        f.write(label+" "+str({k:round(float(np.mean([r[k] for r in rs])),4) for k in rs[0]})+"\n")
    f.write(f"Vertical dose slope={zlo@zhi/(zlo@zlo):.3f}; correlation={np.corrcoef(zlo,zhi)[0,1]:.3f}.\n")
```
