I’ll save the action readout and clear every diagnostic edit. I’ll also correct a swapped pair in the request ordering using the recorded input paths, then inspect the projection’s conditioning so it does not rely on unstable, nearly overlapping contrast and protected directions.
```python
z21=collect_weighted_readout(21)
# Correct the two exchanged request entries, verified against tool-recorded input paths.
rr=json.loads((W/"probe_21/results.json").read_text())
src_to_i={str(Path(r["input"]).resolve()):r["idx"] for r in rr}
order=[src_to_i[str((W/"x"/f"{i:03d}").resolve())] for i in range(128)]
assert sorted(order)==list(range(128))
z21=z21[order]
np.save(W/"weighted_action_0.npy",z21)
act_spec=torch.load(W/"active_readout.pt",weights_only=False)
HA=torch.zeros(128,13,1024,dtype=torch.float64)
HA[:,:,act_spec["feature_ids"]]=torch.from_numpy(z21)
torch.save({"features":HA,"feature_ids":act_spec["feature_ids"]},W/"weighted_action_features.pt")
clear_edits()
print("DIAGNOSTIC EDITS CLEARED",len(list(edited())),"reordered",[(i,j) for i,j in enumerate(order) if i!=j])

def orth(A, rel=1e-8):
    U,s,V=torch.linalg.svd(A,full_matrices=False)
    return U[:,s>s[0]*rel]
def prepare_weighted(expert, HH, ids, contrast_rank=128, clean_energy=.9999, min_angle=.15):
    dim=HH.shape[-1]
    w0=load_original(f"mot.mixtures.{expert}.text_embedding.0.weight").double()
    b0=load_original(f"mot.mixtures.{expert}.text_embedding.0.bias").double()
    excluded=[i for i in range(dim) if i not in set(ids.tolist())]
    fixed=[HH[48:,0].T, w0@Qp, b0[:,None], torch.eye(dim,dtype=torch.float64)[:,excluded]]
    if expert=="video":
        fixed.append(un["C"])
    C0=orth(torch.cat(fixed,dim=1))
    scale=HH[48:].square().mean((0,2)).sqrt().clamp_min(1e-6)
    clean=(HH[48:]/scale[None,:,None]).reshape(-1,dim).T
    clean-=C0@(C0.T@clean)
    U,s,_=torch.linalg.svd(clean,full_matrices=False)
    k=int(torch.searchsorted(s.square().cumsum(0)/s.square().sum(),clean_energy))+1
    C=torch.cat([C0,U[:,:k]],dim=1)
    X=((HH[:48]-HH[48:96])/scale[None,:,None]).reshape(-1,dim).T
    U,s,_=torch.linalg.svd(X,full_matrices=False)
    B0=U[:,:contrast_rank]
    Z=B0-C@(C.T@B0)
    Uz,angles,Vz=torch.linalg.svd(Z,full_matrices=False)
    selected=angles>=min_angle
    B=B0@Vz[selected].T
    D=Uz[:,selected]/angles[selected][None,:]
    w2=load_original(f"mot.mixtures.{expert}.text_embedding.2.weight").double()
    delta=(w2@B)@D.T
    out={"B":B,"D":D,"C":C,"angles":angles,"contrast_rank":contrast_rank,
         "clean_energy":clean_energy,"min_angle":min_angle,"expert":expert}
    print(expert,"C",C.shape[1],"B initial/retained",contrast_rank,B.shape[1],
          "angles_quantiles",torch.quantile(angles,torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).round(decimals=4).tolist(),
          "relative_change",round(float(delta.norm()/w2.norm()),4),
          "protected_error",float((D.T@C).abs().max()))
    return out
wv=prepare_weighted("video",weighted,spec["feature_ids"],contrast_rank=128)
wa=prepare_weighted("action",HA,act_spec["feature_ids"],contrast_rank=64)
torch.save(wv,W/"weighted_video_projector.pt")
torch.save(wa,W/"weighted_action_projector.pt")
with (W/"notes.md").open("a") as f:
    f.write("\nPasses18-20 video weighted readouts complete: weighted_video_features.pt. Pass21 action weighted readout complete: weighted_action_features.pt; swapped request positions42/43 corrected using registered input paths. All175 diagnostic edits cleared. Projector factors saved weighted_{video,action}_projector.pt. Based only on static SVD/principal angles, not action-target fitting.\n")
```
