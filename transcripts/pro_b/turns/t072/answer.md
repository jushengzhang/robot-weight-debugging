The fast-RoPE edit did not improve success and increased re-grasp-loop failures, so I will revert it. I will now focus on environment failures at the visual input stage: table changes may introduce a large low-frequency color shift before object features are formed. I will compare camera color statistics across saved perturbations and inspect the patch embedding’s sensitivity to uniform color.
```python
for n in list(edited()): drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 23: fast-RoPE edit rejected\n"
            "60/128->60/128 (0.0 +/-1.6pp); position20->21/68, task4->3/15; "
            "loop failures33->39. Reverted. No support for this position-bias repair.\n")
# Image statistics only; no model computation in the kernel.
imgstats={}
groups={}
for r in r1:
    p=Path(r["frames_dir"])/"t0000_inputs.pt"
    d=torch.load(p,map_location="cpu",weights_only=False)["element"]["observation"]
    stats={}
    for v,k in [("base","image"),("wrist","wrist_image")]:
        im=np.asarray(d[k],dtype=np.float64)/127.5-1.0
        mu=im.mean((0,1))
        edge=(np.abs(np.diff(im,axis=0)).mean()+np.abs(np.diff(im,axis=1)).mean())/2
        stats[v]=(mu,edge)
    imgstats[r["id"]]=stats
    e=byid[r["id"]]
    groups.setdefault((e["suite"],e["task"],e["init_idx"]),{})[e["ptype"]]=r["id"]
color_deltas={p:[] for p in ["object","position","environment","task"]}
for g in groups.values():
    if "semantic" not in g: continue
    for p in color_deltas:
        if p not in g: continue
        for v in ["base","wrist"]:
            color_deltas[p].append(imgstats[g[p]][v][0]-imgstats[g["semantic"]][v][0])
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
v=torch.tensor(color_u,dtype=torch.float32)[:,None,None].expand(3,14,14).reshape(-1)/14
Pflat=PW.reshape(PW.shape[0],-1)
print("DC rank1 fraction of patch weight energy",round(float((Pflat@v).square().sum()/Pflat.square().sum()),4))
canonical_rgb=np.stack([s[v][0] for i,s in imgstats.items() if byid[i]["ptype"]=="semantic"
                        for v in ["base","wrist"]]).mean(0)
print("Canonical normalized RGB mean",np.round(canonical_rgb,4).tolist())
```
