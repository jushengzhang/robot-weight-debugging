The recovered time coefficients predict unused gate measurements to within about 0.3–0.8%, so they are accurate enough for this analysis. I will use the final normalization’s scale to recompute action-sensitive directions and determine where observation-dependent gripper decisions first appear, separating them from the fixed initial action noise.
```python
gamma=1+finalmod[:,:1024]
eff=gamma[:,None,:]*out[None,:7,:]
unit_eff=eff/eff.norm(dim=-1,keepdim=True)
print("Cosine of effective vs raw action readout, first/last call:")
print(np.round(torch.einsum("tad,ad->ta",unit_eff,axes)[[0,9]].numpy(),3).tolist())
ranking2=[]
for L in range(1,18):
    op=load_original(f"{expert}layers.{L}.self_attn.o_proj.weight").reshape(1024,8,256)
    gg=torch.from_numpy(gates[f"L{L}_ga"]).float()
    cc=torch.einsum("tad,td,dhv->tahv",unit_eff,gg,op)
    for h in range(8):
        ge=cc[:,6,h].square().sum(-1).mean().item()
        pe=cc[:,:6,h].square().sum(-1).mean().item()
        ranking2.append((ge,L,h,ge/(pe+1e-12)))
print("Top effective gripper-coupled heads (energy, layer, head, preference):")
for z in sorted(ranking2,reverse=True)[:20]: print(tuple(round(q,4) if isinstance(q,float) else q for q in z))
for k in [4,6,12,13,15,23]:
    z=np.load(W/f"probe_{k}"/"probe_orig_0_acts.npz")
    print("probe",k,"keys",z.files[:5],"...",z.files[-3:])
# Residual contribution, grouping by the saved initial noise to remove its confound.
j15=json.loads((W/"probe_15"/"results.json").read_text())
r15=[]; acts15=[]; ng=[]
for rr in j15:
    z=np.load(rr["activations"])
    ls=[int(key.split(".")[1]) for key in z.files if key.startswith("calls::expert.") and ".resid::" in key]
    ls=sorted(ls)
    r15.append(np.stack([z[f"calls::expert.{L}.resid::action"] for L in ls]))
    acts15.append(np.load(rr["action"])[:,6].mean())
    pp=torch.load(rr["input"],weights_only=False,map_location="cpu")
    ng.append(np.asarray(pp["noise"]).tobytes())
r15=np.stack(r15)
gp=np.array([0 if b==ng[0] else 1 for b in ng])
q=unit_eff[0,6].numpy()
inc=np.diff(r15[:,:,0,:],axis=1)
yp=np.asarray(acts15).copy()
proj=inc@q
for g in np.unique(gp):
    ii=gp==g
    yp[ii]-=yp[ii].mean()
    proj[ii]-=proj[ii].mean(0)
print("Observation-dependent first-call grip increment: layer std correlation(final grip)")
for j,L in enumerate(ls[1:]):
    co=np.corrcoef(proj[:,j],yp)[0,1]
    print(L,round(proj[:,j].std(),3),round(co,3))
```
