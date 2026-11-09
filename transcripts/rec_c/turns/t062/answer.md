I will summarize camera attention mass, spatial concentration, and action changes from that probe. This checks the intervention's immediate effect; recovery benefit still requires paired rollouts.
```python
r23=json.loads((W/"probe_23/results.json").read_text())
g23=defaultdict(dict)
for r in r23:g23[r["idx"]][r["arm"]]=r
for i,rs in sorted(g23.items()):
    a,b=rs["orig"],rs["edit"]
    aa,bb=np.load(a["action"]),np.load(b["action"])
    print("query",i,"XYZ RMS",round(float(np.sqrt(np.mean((aa[:,:3]-bb[:,:3])**2))),4),
          "xyz",np.round(aa[:5,:3].mean(0),3).tolist(),"->",np.round(bb[:5,:3].mean(0),3).tolist(),
          "grip",round(float(aa[:5,6].mean()),3),"->",round(float(bb[:5,6].mean()),3))
    for L,hs in heads.items():
        ma,mb=np.load(a["attention"][str(L)]).mean(1),np.load(b["attention"][str(L)]).mean(1)
        for h in hs:
            va,vb=ma[h],mb[h]
            def met(v):
                im=v[:512];p=im/(im.sum()+1e-12)
                eff=np.exp(-(p*np.log(p+1e-12)).sum())
                return [v[:256].sum(),v[256:512].sum(),v[768:].sum(),1-v.sum(),eff]
            print(" L%dH%d"% (L,h),np.round(met(va),3).tolist(),"->",np.round(met(vb),3).tolist())
```
