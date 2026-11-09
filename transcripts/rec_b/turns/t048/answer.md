I will compare each MLP’s norm and its projection onto the gripper readout, separately at the start and later in the episode. I will also check the recorded robot motion and image change, because a gripper signal is only suspicious if it accompanies repeated commands without useful movement.
```python
meta18=json.loads((W/"probe_18"/"results.json").read_text())
def probe_identity(r):
    s=str(localpath(r["input"]))
    m=re.search(r"/s(\d+)_(open|close)_seed(\d+)_(original|edited)/t(\d+)",s)
    if m:
        return int(m[1]),m[2],int(m[3]),int(m[5])
    # Resolve alternate artifact paths through saved frame aliases.
    target=localpath(r["input"]).resolve()
    for rr in rolls[1]:
        dd=frame_dir(rr)
        pp=dd/target.name
        if pp.exists() and pp.resolve()==target:
            return rr["id"],rr["grip"],rr["seed"],int(target.name[1:5])
    raise ValueError(s)
ident18=[probe_identity(r) for r in meta18]
success18=np.array([bidx[t[:3]]["success"] for t in ident18])
late18=np.array([t[3]>0 for t in ident18])
acts18=[np.load(localpath(r["activations"])) for r in meta18]
M18=np.stack([np.stack([z[f"calls::expert.{l}.mlp_out::action"] for l in range(18)]) for z in acts18])
print("MLP calls shape",M18.shape)
gn=g.numpy()
p18=M18@gn
norm18=np.linalg.norm(M18,axis=-1)
def effect(v,mask):
    f=v[mask & ~success18];s=v[mask & success18]
    return float((f.mean()-s.mean())/np.sqrt((f.var()+s.var())/2+1e-12))
print("L | late grip fail/success | effect late/start | norm fail/success | grip-call0/call9 failure")
for l in range(18):
    v=p18[:,l,:].mean(1);nv=norm18[:,l,:].mean(1)
    fm=late18&~success18;sm=late18&success18
    print(l, *[round(float(q),3) for q in
               [v[fm].mean(),v[sm].mean(),effect(v,late18),effect(v,~late18),
                nv[fm].mean(),nv[sm].mean(),p18[fm,l,0].mean(),p18[fm,l,-1].mean()]])
action18=np.stack([np.load(localpath(r["action"])) for r in meta18])
print("Late predicted grip means fail/success:",
      np.round(action18[late18&~success18,:,6].mean(1),2).tolist(),
      np.round(action18[late18&success18,:,6].mean(1),2).tolist())
print("Late action translation magnitude fail/success",
      np.round(np.linalg.norm(action18[late18&~success18,:,:3],axis=2).mean(1),4).tolist(),
      np.round(np.linalg.norm(action18[late18&success18,:,:3],axis=2).mean(1),4).tolist())
print("Temporal input changes for looping failures:")
for i in matched:
    r=next(r for r in rolls[1] if r["id"]==i and not r["success"])
    dd=frame_dir(r)
    pp=[dd/f"t{t:04d}_inputs.pt" for t in [0,250,500]]
    if not all(p.exists() for p in pp):continue
    obs=[torch.load(p,weights_only=False,map_location="cpu")["element"] for p in pp]
    states=np.stack([q["observation/state"] for q in obs])
    changes=[]
    for k in ["observation/image","observation/wrist_image"]:
        changes.append(round(float(np.abs(obs[-1][k].astype(float)-obs[1][k]).mean()/255),3))
    print(i,r["grip"],"eef_xyz",np.round(states[:,:3],3).tolist(),
          "grip_state",np.round(states[:,-2:],3).tolist(),"img_MAE_250to500",changes)
```
