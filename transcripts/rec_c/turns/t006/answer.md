I will summarize each attention head by camera and instruction mass, concentration, and spatial peak. I will also correct the activation comparison to exclude call-count metadata and inspect the actual gripper/motion traces; these are descriptive checks, not yet evidence that any head causes the loop.

```python
with open(W/"probe_2/results.json") as f:
    pr2=json.load(f)
print("probe structure:", type(pr2).__name__)
print("probe example:", json.dumps(pr2 if not isinstance(pr2,list) else pr2[:1],ensure_ascii=True)[:5500])
print("mean-only layer contrasts:")
for s in sorted([s for s in stats1 if s[0].startswith("mean::")],key=lambda t:-t[1])[:24]:
    print(s[0], *(round(v,4) for v in s[1:]))
print("matched-scene activation contrasts:")
matched_good=[rs for rs in matched1 if not any(r.get("success_at_t0") for r in rs)
              and any(not r["success"] and r["grip_flips"]>=6 for r in rs)]
mstats=[]
for k in a0.files:
    if not k.startswith("mean::"):
        continue
    diffs=[]
    for rs in matched_good:
        s=next(r for r in rs if r["success"])
        l=next(r for r in rs if not r["success"])
        diffs.append(A1[(l["id"],l["grip"])][k]-A1[(s["id"],s["grip"])][k])
    ds=np.stack(diffs)
    scale=np.sqrt(np.mean([np.linalg.norm(A1[(r["id"],r["grip"])][k])**2 for rs in matched_good for r in rs]))
    mstats.append((k,np.linalg.norm(ds.mean(0))/scale, np.mean(np.linalg.norm(ds,axis=-1))/scale))
for s in sorted(mstats,key=lambda x:-x[1])[:14]:
    print(s[0],round(s[1],4),round(s[2],4))
for id_,grip in [(979,"open"),(562,"close"),(562,"open"),(712,"open")]:
    r=next(r for r in valid1 if r["id"]==id_ and r["grip"]==grip)
    a=np.load(r["actions_npy"])
    print("trace", id_,grip,r["instruction"],"success",r["success"])
    for start in [0,50,250,450]:
        b=a[start:start+50]
        if len(b):
            flips=int(np.count_nonzero(np.diff(b[:,6]>0)))
            print(" t",start,"meanXYZ",np.round(b[:,:3].mean(0),4).tolist(),
                  "XYZrms",round(float(np.sqrt(np.mean(b[:,:3]**2))),4),
                  "grip[min,mean,max]",np.round([b[:,6].min(),b[:,6].mean(),b[:,6].max()],3).tolist(),
                  "flips",flips)
print("saved input element fields:", {k:(list(v.shape) if hasattr(v,"shape") else type(v).__name__) for k,v in inp0["element"].items()})
```
