I will rank late heads by their gripper-specific output coupling after the measured adaptive residual gates, and compare that with their attention to action tokens. This avoids the earlier mistake of ranking heads by raw output weights alone.
```python
expert="paligemma_with_expert.gemma_expert.model."
out=load_original("action_out_proj.weight")
axes=out[:7]/out[:7].norm(dim=1,keepdim=True)
ranking=[]
for L,pk in [(15,2),(16,5),(17,3)]:
    attns=[]
    for i in range(4):
        a=np.load(W/f"probe_{pk}"/f"probe_orig_{i}_attn_L{L}.npy")
        if i==0: print("Map shape L",L,a.shape)
        attns.append(a)
    op=load_original(f"{expert}layers.{L}.self_attn.o_proj.weight")
    g=torch.from_numpy(gates[f"L{L}_ga"]).float()
    # Coupling of each head into unit-normalized direct pose/grip readout axes.
    c=torch.einsum("ad,td,dhv->tahv",axes,g,op.reshape(1024,8,256))
    for h in range(8):
        ge=c[:,6,h].square().sum(-1).mean().item()
        pe=c[:,:6,h].square().sum(-1).mean().item()
        row={"L":L,"H":h,"grip":ge,"pose":pe,"ratio":ge/(pe+1e-12)}
        masses=[]
        for a in attns:
            # Maps expose only prefix and action-token source positions.
            ah=a[h] if a.shape[0]==8 else (a[:,h].mean(0) if a.shape[1]==8 else None)
            if ah is not None:
                am=ah.mean(axis=0) if ah.ndim==2 else ah
                masses.append([am[:256].sum(),am[256:512].sum(),am[512:-10].sum(),am[-10:].sum()])
        row["masses"]=np.round(np.mean(masses,axis=0),3).tolist() if masses else []
        ranking.append(row)
for r in sorted(ranking,key=lambda r:r["grip"],reverse=True):
    print(r["L"],r["H"],"grip",round(r["grip"],4),"ratio",round(r["ratio"],2),"B/W/rest/self",r["masses"])
print("Transition manifest structure",brief(json.loads((W/"transition_probe_manifest.json").read_text())))
```
