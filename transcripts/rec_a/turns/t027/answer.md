The second camera-routing edit also hurt recovery, so I will revert it and abandon a simple “more base-camera attention” repair. I will now investigate the missing proprioception problem directly: using the saved robot state only for analysis, I will check whether visually different open/closed gripper starts produce distinguishable internal representations and appropriately different action chunks.
```python
print("Pass 11:",paired_summary(11))
clear_edits()
initial_obs={}
for r in r1:
    fp=Path(r["frames_dir"])/"t0000_inputs.pt"
    if fp.exists():
        initial_obs[(r["id"],r["grip"])]=torch.load(fp,map_location="cpu",weights_only=False)
print("Initial state examples and predicted gripper chunks:")
for i in [367,753,11,1269,817,4]:
    for g in ["open","close"]:
        ob=initial_obs[(i,g)]
        st=np.asarray(ob["element"]["observation/state"])
        ac=np.asarray(ob["action_chunk"])
        print(i,g,"state",ascii_round(st,4),"grip",ascii_round(ac[:,6],2))
print("Same-scene start-pair comparison (exclude t0-solved pairs):")
cmp=[]
for i in diag_ids:
    oo=initial_obs.get((i,"open")); oc=initial_obs.get((i,"close"))
    rr=[r for r in r1 if r["id"]==i]
    if oo is None or oc is None or any(r["success_at_t0"] for r in rr): continue
    so=np.asarray(oo["element"]["observation/state"])
    sc=np.asarray(oc["element"]["observation/state"])
    ao=np.asarray(oo["action_chunk"]); ac=np.asarray(oc["action_chunk"])
    cmp.append([np.linalg.norm(so[:3]-sc[:3]),np.linalg.norm(so[-2:]-sc[-2:]),
                np.sqrt(np.mean((np.asarray(oo["noise"])-np.asarray(oc["noise"]))**2)),
                np.mean(np.sign(ao[:,6]) != np.sign(ac[:,6])),
                np.sqrt(np.mean((ao[:,:3]-ac[:,:3])**2))])
print("n",len(cmp),"median pose/finger/noise/sign-disagreement/XYZ delta",ascii_round(np.median(cmp,axis=0),4))
print("Initial open-minus-close representation contrast consistency on four scene pairs:")
for stream,ls,grp in [("vision",[0,13,26],"wrist"),("vlm",[0,3,6,9,12,15,16],"wrist"),
                     ("vlm",[0,3,6,9,12,15,16],"base"),("expert",[0,3,6,9,12,15,17],"action")]:
    for l in ls:
        k=f"mean::{stream}.{l}.resid::{grp}"
        D=np.stack([res4[a][k]-res4[b][k] for a,b in [(0,1),(4,5),(8,9),(12,13)]])
        mu=D.mean(0)
        norm=np.linalg.norm(D,axis=1)
        consistency=np.linalg.norm(mu)/(np.sqrt((D*D).sum(1).mean())+1e-9)
        print(stream,l,grp,"rmsdiff",round(float(np.sqrt((D*D).mean())),4),"consistency",round(float(consistency),3))
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 10-11: L12 H3 camera-direction removal rejected\n"
            "- Base attention increased as intended; XYZ changes up to 0.017 RMS, gripper timing unchanged.\n"
            "- Screen 31/64 -> 26/64, -7.8 pp (SE 4.0), 1 rescue / 6 losses, loops 21->23.\n"
            "- Reverted. Evidence rejects a simple camera-rebalance repair in these two heads.\n"
            "- Next diagnosis: visually encoded finger state versus action conditioning; saved state used "
            "only as an analysis label, never as policy input.\n")
```
