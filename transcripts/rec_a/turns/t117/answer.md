The rank-one instruction edit also failed (one rescue, five losses), so I will revert it. I will now check a different localization error: whether spatial heads spread attention over several image patches in loops, potentially averaging incompatible grasp locations. This requires examining image-attention concentration, not just how much attention each camera receives.
```python
for n in list(edited()): drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 37: dominant query-mode attenuation rejected\n"
            "- 31/64 -> 27/64, -6.2 pp (SE 3.8), 1 rescue / 5 losses; loops 21 -> 25. Reverted.\n"
            "- Reducing observed word fixation was not sufficient for better recovery.\n")
maps={}
target_paths=[str(pathlib.Path(x["input"]).resolve()) for x in j5[:4]]
for pk in [2,3,5,31]:
    jj=json.loads((W/f"probe_{pk}"/"results.json").read_text())
    for row in jj:
        if row["arm"]!="orig":continue
        path=str(pathlib.Path(row["input"]).resolve())
        if path not in target_paths:continue
        ix=target_paths.index(path)
        for fp in (W/f"probe_{pk}").glob(f"probe_orig_{row['idx']}_attn_L*.npy"):
            L=int(fp.stem.split("_L")[-1])
            maps.setdefault(L,{})[ix]=np.load(fp)
cand=[]
for L,ms in maps.items():
    if len(ms)!=4 or L==0:continue
    op=load_original(f"{expert}layers.{L}.self_attn.o_proj.weight").reshape(1024,8,256)
    gg=torch.from_numpy(gates[f"L{L}_ga"]).float()
    cc=torch.einsum("tad,td,dhv->tahv",unit_eff,gg,op)
    pose=cc[:,:6].square().sum(-1).mean((0,1)).numpy()
    for h in range(8):
        vals=[]
        for i in range(4):
            av=ms[i][h].mean(0)
            im=av[:512]; mass=im.sum(); p=im/(mass+1e-20)
            ent=float(-(p*np.log(p+1e-20)).sum())
            vals.append([float(mass),ent,float(p.max())])
        vals=np.array(vals)
        contrast=(vals[[0,2],1]-vals[[1,3],1])
        if vals[:,0].mean()>.25 and vals[[0,2],0].min()>.1:
            cand.append({"L":L,"H":h,"pose":float(pose[h]),"ent_diff":float(contrast.mean()),
                         "pair_diff":np.round(contrast,3).tolist(),
                         "mass":np.round(vals[:,0],3).tolist(),
                         "peak":np.round(vals[:,2],3).tolist()})
for c in sorted(cand,key=lambda z:z["pose"]*max(z["ent_diff"],0),reverse=True)[:15]:
    print(c)
(W/"current_candidate.json").write_text(json.dumps({"name":"none"}))
```
