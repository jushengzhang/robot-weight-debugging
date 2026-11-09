The full-strength edit repairs four episodes with no observed regressions, versus one repair at half strength. I will check which task substitutions changed, quantify the replay dose response, and save this small edit as a reversible checkpoint before considering additional semantic families.
```python
d8=probe_dict(8)
for i in [0,2]:
    ref=np.load(p5[i+1]["action"])[:,:6]
    results=[]
    for label,p in [("zero",d6[i,"orig"]),("half",d6[i,"edit"]),("full",d8[i,"edit"])]:
        aa=np.load(p["action"])[:,:6]
        results.append((label,round(float(np.linalg.norm(aa-ref)),4),np.round(aa.mean(0),3).tolist()))
    print("dose response case",i,results)
r9=read_roll(9)
def paired_summary(rs, print_changes=False):
    pairs=collections.defaultdict(dict)
    for r in rs:
        pairs[(r["suite"],r["id"],r["seed"])][r["model"]]=r
    diffs=[];valid_diffs=[];changes=[]
    for key,rr in pairs.items():
        if not {"original","edited"}<=rr.keys(): continue
        o,e=rr["original"],rr["edited"]
        d=int(e["success"])-int(o["success"])
        diffs.append(d)
        if o.get("termination") is not None and e.get("termination") is not None:
            valid_diffs.append(d)
        if d:
            changes.append((d,key,o.get("instruction") or e.get("instruction")))
    def stat(ds):
        a=np.array(ds,dtype=float)
        return {"n":len(a),"diff_pp":round(a.mean()*100,2),"se_pp":round(a.std(ddof=1)/np.sqrt(len(a))*100,2),
                "repairs":int((a==1).sum()),"regressions":int((a==-1).sum())}
    out={"all":stat(diffs),"valid":stat(valid_diffs)}
    if print_changes:
        for x in changes: print(x)
    return out
print("full roll summary",paired_summary(r9,True))
print("edited files", [(p.name,p.stat().st_size) for p in (W/"edited").iterdir()][:6])
(W/"checkpoints").mkdir(exist_ok=True)
shutil.copytree(W/"edited",W/"checkpoints"/"lex_small_full",dirs_exist_ok=True)
print("checkpoint saved")
# Distinguish likely incorrect task in the two diagnostic initial action chunks.
for i in [0,2,5,7]:
    aa=np.load(p5[i]["action"])[:,:6]
    refs=[j for j in [1,3,4,6,8,9] if j!=i]
    ranked=sorted([(float(np.linalg.norm(aa-np.load(p5[j]["action"])[:,:6])),j) for j in refs])
    print("original case",i,"nearest first actions",[(j,round(d,3)) for d,j in ranked[:3]])
with (W/"notes.md").open("a") as f:
    f.write("\nPass 9 full 5-row edit: 4 repairs/0 regressions; +6.25 pp, SE 3.0 on n64 "
            "(16 matched nulls). Snapshot checkpoints/lex_small_full. "
            "No changes outside the embedding table. Next: independent validation and cautiously add "
            "object-specific descriptive anchors, not ambiguous dish/container/appliance rows.\n")
```
