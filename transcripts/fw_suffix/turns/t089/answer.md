I’ll remove all temporary edits and check the anchor-corrected readouts. If they vary substantially across denoising steps despite a fixed text context, I won’t use them to construct a repair.
```python
for n in list(edited()):
    drop_edit(n)
print("Instrumentation removed; edited tensor count:",len(edited()))
z0=np.load(W/"probe_30/probe_edit_0_acts.npz")
print("Activation keys:",z0.files[:6])
read_layers = {}
anchor_ranges=[]
for l in range(1,9):
    all_calls=[]
    for i in range(80):
        z=np.load(W/f"probe_30/probe_edit_{i}_acts.npz")
        k=f"calls::action.{l}.attn_out::action"
        if k not in z:
            k=[k for k in z.files if k.startswith("calls::") and f".{l}." in k][0]
        a=z[k][:,:AR].astype(np.float64)
        denom=a[:,1:2]-a[:,0:1]
        all_calls.append((a[:,2:]-a[:,0:1])/denom)
        anchor_ranges.append((l,float(denom.min()),float(denom.max())))
    read_layers[l]=np.stack(all_calls)
    x=read_layers[l]
    print("layer",l,"rms",round(float(np.sqrt(np.mean(x*x))),6),
          "time_std_rms",round(float(np.sqrt(np.mean(np.var(x,axis=1)))),7),
          "finite",bool(np.isfinite(x).all()))

Acalls=np.zeros((80,10,1024),dtype=np.float64)
Acalls[:,:,af[:512]]=read_layers[1]
Acalls[:,:,af[512:]]=(read_layers[2]-read_layers[1])[:,:,:511]
Vcalls=np.zeros((80,10,3072),dtype=np.float64)
vfirst=np.concatenate([read_layers[l] for l in (3,4,5)],axis=2)
vsum=np.concatenate([read_layers[l] for l in (6,7,8)],axis=2)
Vcalls[:,:,vf[:1536]]=vfirst
Vcalls[:,:,vf[1536:]]=(vsum-vfirst)[:,:,:1535]
pc={}
for ex,x in (("action",Acalls),("video",Vcalls)):
    M=torch.from_numpy(x.mean(axis=1)).float()
    pc[ex]=M
    np.save(W/f"masked_text_{ex}_80.npy",M.numpy())
    Dp=M[:48:2]-M[1:48:2]
    s=torch.linalg.svdvals(Dp)
    print("\n",ex,"mean norm clean/raw",
          round(float(M[clean_idx].norm(dim=1).mean()),5),
          round(float(M[raw_idx].norm(dim=1).mean()),5),
          "delta_norm",round(float(Dp.norm(dim=1).mean()),5),
          "top_singular",np.round(s[:8].numpy(),5))
    for cat in sorted(set(cats)):
        ids=[i for i,c in enumerate(cats) if c==cat]
        print(cat,round(float(Dp[ids].norm(dim=1).mean()),5))
    print("Recovered time variation RMS:",
          round(float(np.sqrt(np.mean(np.var(x,axis=1)))),7))
print("Anchor denominator min/max:",
      min(t[1] for t in anchor_ranges),max(t[2] for t in anchor_ranges))
torch.save({"contexts":pc,"action_features":af,"video_features":vf},
           W/"masked_text_contexts.pt")
(W/"current_edit.json").write_text(json.dumps({"name":"none","edited_count":0}))
```
