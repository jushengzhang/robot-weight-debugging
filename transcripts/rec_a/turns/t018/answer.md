Halving this head did not help: success fell by 3.1 points (paired SE 3.8), and loop failures increased from 21 to 24. I will inspect the exact action changes and reversals, then revert the ablation. This result weakens the idea that the head is simply an excessive grasp prior; it may instead stabilize a task-consistent decision.
```python
p6=json.loads((W/"probe_6/results.json").read_text())
print("Probe arms:",collections.Counter(p["arm"] for p in p6))
for ii in range(4):
    zz=[p for p in p6 if p["idx"]==ii]
    aa={p["arm"]:np.load(p["action"]) for p in zz}
    if len(aa)==2:
        ao=aa.get("orig",aa.get("original")); ae=aa.get("edit",aa.get("edited"))
        if ae is None: ae=aa[[a for a in aa if a!="orig" and a!="original"][0]]
        print(ii,"original/edited mean grip",ascii_round([ao[:,6].mean(),ae[:,6].mean()]),
              "xyz rms diff",round(float(np.sqrt(np.mean((ao[:,:3]-ae[:,:3])**2))),4),
              "rot rms diff",round(float(np.sqrt(np.mean((ao[:,3:6]-ae[:,3:6])**2))),4),
              "grip sign changes",int(np.sum((ao[:,6]>0)!=(ae[:,6]>0))))
def pair_records(passnum):
    rr=read_jsonl(W/f"roll_{passnum}.jsonl")
    d={}
    for r in rr:
        key=(r["id"],r["grip"],r["seed"])
        d.setdefault(key,{})[r["model"]]=r
    return [(k,x.get("original"),x.get("edited")) for k,x in d.items() if "original" in x and "edited" in x]
def paired_summary(passnum, allowed_ids=None):
    pairs=pair_records(passnum)
    if allowed_ids is not None:
        pairs=[x for x in pairs if x[0][0] in allowed_ids]
    d=np.array([int(e["success"])-int(o["success"]) for k,o,e in pairs],dtype=float)
    return {"n":len(d),"orig":sum(o["success"] for _,o,e in pairs),
            "edit":sum(e["success"] for _,o,e in pairs),"diff_pp":float(100*d.mean()) if len(d) else None,
            "se_pp":float(100*d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else None,
            "rescued":sum(d==1),"lost":sum(d==-1),
            "loops_o":sum(not o["success"] and o["grip_flips"]>=6 for _,o,e in pairs),
            "loops_e":sum(not e["success"] and e["grip_flips"]>=6 for _,o,e in pairs)}
print("Pass 7:",paired_summary(7))
for key,o,e in pair_records(7):
    if o["success"]!=e["success"]:
        print("Changed",key,o["success"],"->",e["success"],
              "flips",o["grip_flips"],"->",e["grip_flips"],o["instruction"])
clear_edits()
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 6-7: E16H3_O_0.5 rejected\n"
            "- Paired screen: 31/64 -> 29/64, -3.1 pp (SE 3.8), 2 rescues / 4 losses.\n"
            "- Loop failures 21 -> 24; mean flips 12.52 -> 14.98.\n"
            "- Contradicts simple excessive-prior interpretation; reverted.\n")
print("Current edits:",edited())
```
