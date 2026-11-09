The 640-episode validation gained only three successes—far too small to distinguish from noise—so I will remove the illumination edit. No candidate has yet met the retention-and-gain standard. With the remaining budget, I will examine one unresolved observation: late expert heads that attend almost exclusively to the action tokens may reinforce a motion pattern during re-grasp loops.
```python
def paired_rows(passes):
    out=[]
    for k in passes:
        rows=[json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines()]
        pp=defaultdict(dict)
        for r in rows:pp[(r["id"],r["seed"])][r["model"]]=r
        out.extend((k,key,v["original"],v["edited"]) for key,v in pp.items() if len(v)==2)
    return out
def report_pairs(ps):
    groups={"all":ps}
    groups.update({t:[p for p in ps if p[2]["ptype"]==t] for t in sorted({p[2]["ptype"] for p in ps})})
    for g,ss in groups.items():
        d=np.array([int(e["success"])-int(o["success"]) for _,_,o,e in ss])
        print(g,"n",len(d),"original",sum(o["success"] for _,_,o,e in ss),"edited",sum(e["success"] for _,_,o,e in ss),
              "delta_pp",round(100*d.mean(),2),"SE_pp",round(100*d.std(ddof=1)/len(d)**.5,2),
              "gains/losses",int((d==1).sum()),int((d==-1).sum()))
psD=paired_rows(range(26,31))
report_pairs(psD)
for n in list(edited()):drop_edit(n)
print("saved diagnostic loop examples")
for r in sorted(r1,key=lambda r:r["grip_flips"],reverse=True)[:12]:
    d=pathlib.Path(r["frames_dir"])
    inputs=sorted(d.glob("*inputs.pt"))
    print(r["id"],r["ptype"],"success",r["success"],"flips",r["grip_flips"],
          r["task"][:45],"inputs",[(p.name) for p in inputs],str(d).replace(str(W),"WORK"))
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes26-30: D0.5 fails independent640 validation\nOriginal379/640 vs edited382/640, +0.47pp (paired SE~0.95); gains20/losses17. Environment80->81/128, object109->111, position35->33, semantic122->124, task33->33. No significant gain; D removed. NO validated edit currently. Remaining10 passes: investigate late action-self-attention heads suggested by existing L16 maps, then test/validate only if clear signal. Otherwise submit unmodified weights rather than unsupported damage.\n")
```
