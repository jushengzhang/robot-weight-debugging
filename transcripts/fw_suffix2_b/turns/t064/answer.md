I will measure the combined candidate against the projection-only result and save it separately. I will also summarize the paired development outcomes so later choices do not rely on probe distances alone.
```python
m24=probe_metrics(24)
print(json.dumps(m24,indent=2))
snapshot_candidate("hybrid_r256_d1_t05_g128")
def paired_rolls(ks):
    pairs=[]
    for k in ks:
        rows=[json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines()]
        by={}
        for r in rows:
            key=(r["suite"],r["id"],r["seed"])
            by.setdefault(key,{})[r["model"]]=r
        for key,z in by.items():
            if "original" in z and "edited" in z:
                pairs.append(z)
    def summary(pp):
        ds=np.array([int(p["edited"]["success"])-int(p["original"]["success"]) for p in pp],dtype=float)
        return {"n":len(pp),"original":sum(p["original"]["success"] for p in pp),
                "edited":sum(p["edited"]["success"] for p in pp),
                "gains":int(np.sum(ds==1)),"losses":int(np.sum(ds==-1)),
                "diff_pp":float(ds.mean()*100),"se_pp":float(ds.std(ddof=1)/np.sqrt(len(ds))*100)}
    ans=summary(pairs)
    ans["by_category"]={c:summary([p for p in pairs if p["original"]["category"]==c])
        for c in sorted(set(p["original"]["category"] for p in pairs))}
    return ans
print("DEVELOPMENT RAW",json.dumps(paired_rolls([22])))
print("DEVELOPMENT STRIP",json.dumps(paired_rolls([23])))
```
