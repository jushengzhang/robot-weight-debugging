I will build matched contrasts from scenes with one successful start and one looping start. Balancing which gripper start succeeds helps cancel the saved-noise confound. Probing both initial and later observations will also distinguish an initial-state difference from a feature that persists during the loop.
```python
rows = readroll(1) + readroll(4)
by_scene=defaultdict(dict)
for r in rows:
    if r["model"]=="original":
        by_scene[r["id"]][r["grip"]]=r
pairs_by_grip=defaultdict(list)
for sid,rs in by_scene.items():
    if set(rs)!= {"open","close"}:
        continue
    good=[r for r in rs.values() if r["success"] and not r["success_at_t0"] and r["steps"]>55]
    bad=[r for r in rs.values() if not r["success"] and r["grip_flips"]>=6]
    if len(good)==1 and len(bad)==1:
        pairs_by_grip[good[0]["grip"]].append((sid,good[0],bad[0]))
for g,ps in pairs_by_grip.items():
    print("success",g,"eligible",len(ps),"IDs",[p[0] for p in ps])
# Equal counts remove the fixed open-vs-close saved-noise direction from the mean contrast.
nside=min(8,*[len(ps) for ps in pairs_by_grip.values()])
selected=[]
for g in ["open","close"]:
    ps=sorted(pairs_by_grip[g],key=lambda p:(p[1]["suite"],p[0]))
    if len(ps)>nside:
        inds=np.linspace(0,len(ps)-1,nside).round().astype(int)
        ps=[ps[i] for i in inds]
    selected+=ps
d=W/"d19";d.mkdir(exist_ok=True)
manifest=[]
for sid,good,bad in selected:
    for phase in ["initial","later"]:
        for role,r in [("success",good),("loop",bad)]:
            src=Path(r["frames_dir"])
            ts=sorted(src.glob("*_inputs.pt"))
            target=0 if phase=="initial" else (50 if role=="success" else 250)
            source=min(ts,key=lambda p:abs(int(p.stem[1:5])-target))
            dest=d/f"{len(manifest):02d}.pt"
            if dest.exists() or dest.is_symlink(): dest.unlink()
            dest.symlink_to(source)
            manifest.append({"id":sid,"grip":r["grip"],"role":role,"phase":phase,
                             "step":int(source.stem[1:5]),"input":str(dest),
                             "source":str(source),"suite":r["suite"],"level":r["level"]})
(W/"mlp_diagnostic_manifest.json").write_text(json.dumps(manifest,indent=2))
req={"type":"probe","model":"original","inputs":[r["input"] for r in manifest],
     "return":{"layers":list(range(18)),"vlm_layers":[0,1,2,4,6,8,10,12,14,15,16,17],"what":"mlp_out"}}
print("matched scenes",len(selected),"inputs",len(manifest),"suites",dict(Counter(p[1]["suite"] for p in selected)))
print(json.dumps(req,separators=(",",":")))
```
