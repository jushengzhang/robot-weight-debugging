The first intervention did not improve success: two rescues were offset by two regressions, so I will not keep it. I will inspect its attention changes and collect token-level residuals to distinguish an unhelpful wrist/gripper feature from useful object-location information; simply sharpening a base-looking head is insufficient.
```python
p5=json.loads((W/"probe_5/results.json").read_text())
def paired_probe(pp):
    d=collections.defaultdict(dict)
    for p in pp:d[p["idx"]][p["arm"]]=p
    return d
for idx,d in paired_probe(p5).items():
    print("input",idx)
    for arm,p in d.items():
        m,gr,ent=attn_stats(np.load(p["attention"]["7"]))
        a=np.load(p["action"])
        print(arm,"L7h7 B/W/L/self",np.round(np.r_[gr[7,[0,1,3]],1-gr[7].sum()],3).tolist(),
              "action first5 mean",a[:5].mean(0).round(3).tolist())
r6=read_roll(6)
def pair_records(rr):
    d=collections.defaultdict(dict)
    for r in rr:d[r["id"],r["grip"],r["seed"]][r["model"]]=r
    return d
print("changed outcomes",[(key,{arm:r["success"] for arm,r in d.items()})
                         for key,d in pair_records(r6).items()
                         if len(d)==2 and d["original"]["success"]!=d["edited"]["success"]])
reset_edits()
# Probe a fixed mix of loop and success inputs; exact model input, no intervention on observations.
resid_inputs=[]
for sid,g,step in [(647,"close",0),(647,"open",0),(1300,"close",0),(1300,"open",0),
                  (647,"close",250),(1102,"open",250),(815,"open",250),(322,"open",250),
                  (487,"open",0),(1158,"open",0),(1340,"close",0),(992,"open",0),
                  (1102,"close",50),(1340,"close",50),(620,"close",50),(1274,"open",50)]:
    p=pathlib.Path(R[sid,g]["frames_dir"])/f"t{step:04d}_inputs.pt"
    if p.exists():resid_inputs.append(str(p))
print("residual inputs:",[str(pathlib.Path(p).relative_to(W)) for p in resid_inputs])
with (W/"notes.md").open("a") as f:
    f.write("\n## Rejected candidate A (passes 5-6)\nQ L7h7 x1.3, 48 paired tuning episodes: "
            "26/48 ->26/48, 2 rescues/2 regressions, 0.0 +/-4.2 pp SE; loops21->19. "
            "Not retained. Base-head sharpening alone is insufficient. Edits reset.\n")
```
