I will measure the replayed action changes and downstream attention shifts before spending a full rollout pass. In particular, I will check whether the change is specific to the candidate pathway and whether the already-successful query is disturbed.

```python
pr5=json.loads((W/"probe_5/results.json").read_text())
def paired_probe_summary(ps, heads=None):
    groups=collections.defaultdict(dict)
    for p in ps:
        groups[p["idx"]][p["arm"]]=p
    out=[]
    for i,rs in sorted(groups.items()):
        if "orig" not in rs or "edit" not in rs:
            print("arms available",i,list(rs))
            continue
        a=np.load(rs["orig"]["action"]); b=np.load(rs["edit"]["action"])
        rec={"idx":i,"xyz_rms":float(np.sqrt(np.mean((a[:5,:3]-b[:5,:3])**2))),
             "rot_rms":float(np.sqrt(np.mean((a[:5,3:6]-b[:5,3:6])**2))),
             "grip_sign_changes":int(((a[:5,6]>0)!=(b[:5,6]>0)).sum())}
        out.append(rec)
        print("query",i,"orig xyz",",".join(f"{x:.3f}" for x in a[:5,:3].mean(0)),
              "edit xyz",",".join(f"{x:.3f}" for x in b[:5,:3].mean(0)),
              "grip",f"{a[:5,6].mean():.3f}->{b[:5,6].mean():.3f}",
              "xyz RMS",f"{rec['xyz_rms']:.3f}")
        for ls in rs["orig"].get("attention",{}):
            aa=np.load(rs["orig"]["attention"][ls]).astype(np.float64).mean(1)
            bb=np.load(rs["edit"]["attention"][ls]).astype(np.float64).mean(1)
            hs=range(8) if heads is None else heads.get(int(ls),[])
            for h in hs:
                m=lambda v:[v[:256].sum(),v[256:512].sum(),v[768:].sum(),1-v.sum()]
                av=m(aa[h]);bv=m(bb[h])
                print(f" L{ls}H{h} base/wrist/lang/self",
                      "/".join(f"{v:.3f}" for v in av),"->","/".join(f"{v:.3f}" for v in bv),
                      f"peak {aa[h].argmax()}->{bb[h].argmax()}")
    return out
probe5_summary=paired_probe_summary(pr5,{2:[7],8:[2,7],14:[1,4],15:[2,6,7]})
print("arms:",sorted({r["arm"] for r in pr5}))
```
