Layer 17 head 1 still reads substantial wrist-camera information, whereas several neighboring heads are almost entirely self-attending. I will construct a visual empty-hand direction for that head, protecting holding/open, language, base-camera, and action-self-value means. The proposed output addition will point toward opening, with its direct pose coupling removed.
```python
VLM="paligemma_with_expert.paligemma.model.language_model."
# Project saved pooled residuals for a value-space feature analysis.
n17=load_original(VLM+"layers.17.input_layernorm.weight")
V17=load_original(VLM+"layers.17.self_attn.v_proj.weight")
def pooled_vlm_values(segment):
    xx=torch.stack([torch.from_numpy(np.load(r["activations"])[f"mean::vlm.16.resid::{segment}"]).float() for r in rr26])
    xn=xx/(xx.square().mean(-1,keepdim=True)+1e-6).sqrt()
    return (xn*(1+n17))@V17.T
VW,VB,VL=[pooled_vlm_values(s) for s in ["wrist","base","lang"]]
X16=torch.stack([torch.from_numpy(np.load(r["activations"])["calls::expert.16.resid::action"]).float() for r in rr26])
in17=C@load_original(E+"layers.17.input_layernorm.dense.weight").T+load_original(E+"layers.17.input_layernorm.dense.bias")
s17,h17,_=in17.chunk(3,-1)
XS=X16/(X16.square().mean(-1,keepdim=True)+1e-6).sqrt()*(1+s17[None])+h17[None]
VS=XS@load_original(E+"layers.17.self_attn.v_proj.weight").T
pv=[]
for gg in ["fully_closed","holding_like","open"]:
    ii=(groups==gg)&derive
    pv.extend([VB[ii].mean(0)[:,None],VL[ii].mean(0)[:,None],VS[ii].mean(0).T])
for gg in ["holding_like","open"]:
    for grip in ["open","close"]:
        ii=(groups==gg)&derive&np.array([p["grip"]==grip for p in phys])
        if ii.sum(): pv.append(VW[ii].mean(0)[:,None])
Pv=torch.cat(pv,1)
uv,sv,_=torch.linalg.svd(Pv,full_matrices=False)
Pv=uv[:,sv>sv.max()*1e-5]
dv0=VW[(groups=="fully_closed")&derive].mean(0)-VW[(groups=="holding_like")&derive].mean(0)
dv=dv0-Pv@(Pv.T@dv0); dv/=dv.norm()
vscore=(VW@dv).numpy()
vscale=float(vscore[(groups=="fully_closed")&derive].mean())
print("Value cue held-out means:",[(gg,round(float(vscore[(groups==gg)&~derive].mean()/vscale),3)) for gg in ["fully_closed","holding_like","open"]])
ve=vscore[(groups=="fully_closed")&~derive]; vo=vscore[(groups!="fully_closed")&~derive]
print("Held-out pairwise separation:",float((ve[:,None]>vo).mean()))
# Approximate readout sensitivities of the recorded residual means, including adaptive gates.
rms17=(X17.square().mean(-1)+1e-6).sqrt().mean(0)
RG=Wo[:7][None,:,:]*(1+fs[:,None,:])*g17a[:,None,:]/rms17[:,None,None]
poseP=RG[:,:6].reshape(-1,1024).T
ug,sg,_=torch.linalg.svd(poseP,full_matrices=False)
poseP=ug[:,sg>sg.max()*1e-5]
wg=RG[:,6].mean(0); wg=wg-poseP@(poseP.T@wg); wg/=wg.norm()
print("Opening-write direct gains across Euler steps:",np.round((RG[:,6]@wg).numpy(),6).tolist())
print("Pose-coupling residual max:",float((RG[:,:6]@wg).abs().max()))
print("Rank-one beta=2 relative head change:",2/float(O17[:,256:512].norm()))
print("Original old Z write norm:",float(torch.load(W/"empty_cue_retreat_directions.pt",weights_only=False)["write_direction"].norm()))
```
