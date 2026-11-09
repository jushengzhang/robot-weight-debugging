The observation keys are flattened with slashes. I will recompute the statistics using that format and quantify whether environment changes have a distinctive uniform-color component large enough to justify a targeted patch-filter edit.
```python
imgstats={}; groups={}
for r in r1:
    p=Path(r["frames_dir"])/"t0000_inputs.pt"
    el=torch.load(p,map_location="cpu",weights_only=False)["element"]
    stats={}
    for view,k in [("base","image"),("wrist","wrist_image")]:
        im=np.asarray(el["observation/"+k] if "observation/"+k in el else el["observation"][k],
                      dtype=np.float64)/127.5-1.0
        mu=im.mean((0,1))
        edge=(np.abs(np.diff(im,axis=0)).mean()+np.abs(np.diff(im,axis=1)).mean())/2
        stats[view]=(mu,edge)
    imgstats[r["id"]]=stats
    e=byid[r["id"]]
    groups.setdefault((e["suite"],e["task"],e["init_idx"]),{})[e["ptype"]]=r["id"]
color_deltas={p:[] for p in ["object","position","environment","task"]}
for gr in groups.values():
    if "semantic" not in gr: continue
    for p in color_deltas:
        if p not in gr: continue
        for view in ["base","wrist"]:
            color_deltas[p].append(imgstats[gr[p]][view][0]-imgstats[gr["semantic"]][view][0])
print("Uniform camera-color shifts relative to semantic control:")
for p,d in color_deltas.items():
    d=np.stack(d)
    print(p,"nviews",len(d),"mean norm",round(float(np.linalg.norm(d,axis=1).mean()),4),
          "RMS by RGB",[round(float(v),4) for v in np.sqrt((d*d).mean(0))])
D=np.stack(color_deltas["environment"])
_,s,vh=np.linalg.svd(D,full_matrices=False)
color_u=vh[0]
print("Top environment RGB direction",np.round(color_u,3).tolist(),
      "energy fraction",round(float(s[0]**2/(s*s).sum()),3))
patchnames=[n for n in NAMES if "embeddings.patch_embedding" in n]
print("Patch tensors:",[(n,SHAPES[n]) for n in patchnames])
patchw=next(n for n in patchnames if n.endswith("weight"))
patchb=next(n for n in patchnames if n.endswith("bias"))
PW=load_original(patchw)
color_v=torch.tensor(color_u,dtype=torch.float32)[:,None,None].expand(3,14,14).reshape(-1)/14
Pflat=PW.reshape(PW.shape[0],-1)
print("DC rank1 fraction of patch weight energy",round(float((Pflat@color_v).square().sum()/Pflat.square().sum()),4))
canonical_rgb=np.stack([s0[view][0] for i,s0 in imgstats.items() if byid[i]["ptype"]=="semantic"
                        for view in ["base","wrist"]]).mean(0)
print("Canonical normalized RGB mean",np.round(canonical_rgb,4).tolist())
```
