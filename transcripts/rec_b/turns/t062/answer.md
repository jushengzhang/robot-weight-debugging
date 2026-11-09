The strengthened gripper pathway gained one episode, again too little to establish a repair. Before choosing the next intervention, I will consolidate the attention evidence across saved probes and compare outcome changes. This should distinguish a recurring visual-selection problem from a handful of fragile trajectories.
```python
for k in [22,23,24]:
    pp=pairs(k)
    print(k,"rescues",[(o["id"],o["grip"]) for o,e in pp if e["success"] and not o["success"]],
          "regressions",[(o["id"],o["grip"]) for o,e in pp if o["success"] and not e["success"]])
seen=set();maps=[]
for k in [2,3,4,5,9,11,13,14]:
    mp=json.loads((W/f"probe_{k}"/"results.json").read_text())
    for r in mp:
        if r["arm"]!="orig" or "attention" not in r:continue
        ident=probe_identity(r)
        succ=bidx[ident[:3]]["success"]
        group="F" if (not succ and ident[3]>=200) else ("S" if succ and ident[3]>=50 else "early")
        for l,pp in r["attention"].items():
            key=(ident,int(l))
            if key in seen:continue
            seen.add(key)
            a=np.load(localpath(pp)).mean(1)
            vals=np.stack([a[:,:256].sum(1),a[:,256:512].sum(1),a[:,768:].sum(1),
                           1-a.sum(1)],axis=1)
            for h in range(8):
                maps.append({"layer":int(l),"head":h,"identity":ident,"group":group,
                             "masses":vals[h],"max_token":int(a[h].argmax()),
                             "peak":float(a[h].max())})
print("Head summary: layer head [base,wrist,lang,self] F vs S; only high-selectivity heads")
for l in sorted(set(r["layer"] for r in maps)):
    for h in range(8):
        f=[r for r in maps if r["layer"]==l and r["head"]==h and r["group"]=="F"]
        s=[r for r in maps if r["layer"]==l and r["head"]==h and r["group"]=="S"]
        if not f or not s:continue
        fm=np.stack([r["masses"] for r in f]).mean(0);sm=np.stack([r["masses"] for r in s]).mean(0)
        if fm.max()>.75 or np.abs(fm-sm).max()>.2:
            print(l,h,"n",len(f),len(s),"F",np.round(fm,2).tolist(),"S",np.round(sm,2).tolist(),
                  "Fpeaks",[r["max_token"] for r in f])
# Audit candidate H and make its stronger dose.
for n in list(edited()):drop_edit(n)
alpha=1.5
Q=load_original(qn);Qe=Q.clone()
Qe[:256]+=alpha*torch.outer(Q[:256]@gi,gi)
save_edit(qn,Qe)
print("Saved stronger Q2h0 grip-specific dose",alpha)
with (W/"notes.md").open("a") as f:
    f.write("\n## H amplification .5, pass24\n"
            "26/48->27/48,+2.1+/-3.6pp,2 rescues/1 regression,loops21->18. "
            "Small, not established. Test stronger amplification1.5 for dose-response.\n")
```
