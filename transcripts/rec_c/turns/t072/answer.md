I will quantify the residual-channel and action changes from the fixed-input probe. In particular, I want to know whether later layers restore the common offset, which would limit the effect of changing its first writer.
```python
r26=json.loads((W/"probe_26/results.json").read_text())
g=defaultdict(dict)
for r in r26:g[r["idx"]][r["arm"]]=r
for i,rs in sorted(g.items()):
    a,b=rs["orig"],rs["edit"]
    aa,bb=np.load(a["action"]),np.load(b["action"])
    print("query",i,"xyz RMS",round(float(np.sqrt(np.mean((aa[:,:3]-bb[:,:3])**2))),4),
          "xyz",np.round(aa[:5,:3].mean(0),3).tolist(),"->",np.round(bb[:5,:3].mean(0),3).tolist(),
          "grip",round(float(aa[:5,6].mean()),3),"->",round(float(bb[:5,6].mean()),3))
    za,zb=np.load(a["activations"]),np.load(b["activations"])
    for L in [0,2,8,15]:
        vals=[]
        for mod in ["base","wrist"]:
            k=f"mean::vlm.{L}.resid::{mod}"
            x,y=za[k],zb[k]
            vals.append((float(x[674]),float(y[674]),float(x[674]**2/(x@x)),float(y[674]**2/(y@y))))
        print(" L",L,"base/wrist [orig674,edit674,orig fraction,edit fraction]",
              np.round(vals,3).tolist())
    za.close();zb.close()
```
