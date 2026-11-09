The gate identity reconstructs held-input residual updates to about 0.3% relative error. The protected feature is also more selective: reserved empty-closure states average 2.28 versus 0.36 for contact states, while source contacts and open controls are nulled. I will route this feature through the final value projection toward the expert’s vertical readout, using the recovered gates and preserving the other six physical readouts in the linearized weight geometry.
```python
# High-precision orthogonal protection, using only source/control features.
cd64=controls.double()
U64,S64,_=torch.linalg.svd(cd64,full_matrices=False)
Q64=U64[:,S64>S64[0]*1e-5]
d64=E[source_mask].double().mean(0)
d64=d64-Q64@(Q64.T@d64)
d64=d64/d64.norm()
d=d64.float()
print("max protected feature score",round(float((cd64.T@d64).abs().max()),8))
src_score=float((E[source_mask]@d).mean())

# Recover the linear readout routing using measured per-time gates and saved residual RMS.
RMS=[]
for i in src_obs:
    r=np.load(p35[i]["activations"])["calls::expert.17.resid::action"]
    RMS.append(np.sqrt(np.mean(r*r,axis=1)))
rms_t=np.median(np.stack(RMS),axis=0)
g_eff=torch.from_numpy((ga/rms_t[:,None]).mean(0)).float()
final_bias=load_original(f"{EXP}.norm.dense.bias").clone()[:1024]
R=aout[:7].clone()*(1+final_bias)[None,:]*g_eff[None,:]
O=load_original(f"{EXP}.layers.17.self_attn.o_proj.weight").clone()
B=torch.cat([R@O[:,h*256:(h+1)*256] for h in range(8)],dim=0).double()
target=torch.zeros(56,dtype=torch.float64)
for h in [1,7]: target[h*7+2]=-1.0  # Euler integration subtracts predicted velocity.
UU,SS,Vh=torch.linalg.svd(B,full_matrices=False)
vunit=Vh.T@((UU.T@target)/SS)
print("routing condition number",round(float(SS.max()/SS.min()),2),
      "constraint error",round(float((B@vunit-target).norm()),8))
nominal_clearance=0.15
dv=(vunit*(nominal_clearance/src_score)).float()
vname=f"{VLM}.layers.17.self_attn.v_proj.weight"
v_orig=load_original(vname).clone()
v_edit=v_orig+torch.outer(dv,d)
save_edit(vname,v_edit)
print("source feature mean",round(src_score,3),"value-write vector norm",round(float(dv.norm()),3),
      "relative V edit",round(float((v_edit-v_orig).norm()/v_orig.norm()),4))
print("expected protected-source write max",round(float((controls.T@d).abs().max()*dv.norm()),6))
torch.save({"feature_direction":d,"value_direction_per_nominal":dv/nominal_clearance,
            "nominal_clearance":nominal_clearance,"gate":torch.from_numpy(ga).float(),
            "readout_constraints":B.float(),"target":target.float(),
            "source_ids":[p["id"] for j,p in enumerate(pairs) if j in source_indices]},
           W/"clearance_value_edit_design.pt")
cc=W/"candidate_empty_clearance_015";cc.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file():shutil.copy2(p,cc/p.name)
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes35-36: selective empty-closure feature and last-layer gate accounting\n"
            "VLM16 wrist feature projected outside33-dimensional span of14 source contacts,16 open controls, "
            "base/lang centroids and sink674. Reserved empty mean2.284(std1.31), contact0.364(std0.487); "
            "7/8 paired positive. Source empty mean2.915. No success labels used.\n"
            "Expert17 attention/MLP gates recovered by residual-update accounting on4 inputs; "
            "held2 reconstruction error0.29%. Attention gates RMS0.43, mixed signs; ungated routing would be wrong.\n"
            "Candidate rank-one VLM17 v_proj edit. Gate-corrected weight algebra routes feature through "
            "expert17 prefix heads1/7 toward upward action (negative flow Z); "
            "zeros first-order readout in all other physical coordinates/heads. "
            "Uses nominal clearance0.15, must verify actual direction and safety in paired probe.\n")
```
