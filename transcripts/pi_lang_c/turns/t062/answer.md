Lighting shows two positive discordances. Since these prompts should not use edited rows, I will audit their tokens and action traces before attributing that difference to the repair. I will also verify the recorded camera activations are unchanged.
```python
for k in [24,25,26]:
    ps=pair_roll(k)
    print("Safety",k,paired_stats(ps))
    affected=[]
    differences=[]
    for key,p in ps.items():
        s=p["original"]["instruction"]
        ids=set(sp.encode(s)) & set(F["rows"])
        if ids: affected.append((key,[(i,ascii_piece(i)) for i in ids]))
        a,b=p["original"],p["edited"]
        if (a["success"],a["steps"]) != (b["success"],b["steps"]):
            paths=[pathlib.Path(p[m]["actions_npy"]) for m in ["original","edited"]]
            info={}
            if all(q.exists() for q in paths):
                aa,bb=[np.load(q) for q in paths]
                n=min(len(aa),len(bb))
                info={"shapes":[aa.shape,bb.shape],"first5_equal":bool(np.array_equal(aa[:5],bb[:5])),
                      "shared_maxdiff":float(np.abs(aa[:n]-bb[:n]).max())}
            differences.append((key,(a["success"],a["steps"]),(b["success"],b["steps"]),s,info))
    print("Touched prompts",affected[:8],"count",len(affected))
    print("Outcome/step differences",differences[:12],"count",len(differences))
cam_equal=collections.Counter()
for key,p in pair_roll(24).items():
    paths=[pathlib.Path(p[m]["activations"]) for m in ["original","edited"]]
    if all(q.exists() for q in paths):
        aa,bb=[np.load(q) for q in paths]
        for field in aa.files:
            if field.startswith("mean::"):
                cam_equal[(field, bool(np.array_equal(aa[field],bb[field])))]+=1
print("Camera activation mismatches",[(k,v) for k,v in cam_equal.items() if not k[1]])
print("Camera activation checks",sum(cam_equal.values()))
with (W/"notes.md").open("a") as f:
    f.write("\n## Initial safety checks\nPass24 Camera:88/128 both, zero discordances. "
            "Pass25 Robot Initial States:98/128 both. Pass26 Lighting:112->114/128, 2 positive/0 negative discordances. "
            "Auditing token/trace invariance rather than attributing simulator variation to lexical edits.\n")
```
