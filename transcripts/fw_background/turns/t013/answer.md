While waiting, I will measure how much each action attention head overlaps the dominant output directions of its layer, and compare those directions with the instruction-attention path. This provides a collateral-risk reference: removing a direction heavily shared with instruction processing would need stronger evidence than removing a texture-specific direction.

```python
t0=time.time()
action_spectra={}
spectral_summary=[]
for l in range(30):
    o=load_original(pname("action",l,"self_attn.o.weight"))
    c=load_original(pname("action",l,"cross_attn.o.weight"))
    f=load_original(pname("action",l,"ffn.2.weight"))
    ev,u=torch.linalg.eigh(o@o.T)
    idx=torch.arange(len(ev)-1,len(ev)-33,-1)
    vals=ev[idx].clamp_min(0)
    U=u[:,idx]
    shares=(U.T@o).reshape(32,24,128).square().sum(-1)
    cproj=(U.T@c).square().sum(-1)
    fproj=(U.T@f).square().sum(-1)
    action_spectra[l]={
        "output_basis":U.numpy(),"eigenvalues":vals.numpy(),
        "head_shares":shares.numpy(),
        "cross_attention_energy":cproj.numpy(),
        "mlp_energy":fproj.numpy(),
    }
    spectral_summary.append({
        "layer":l,
        "top1_fraction":float(vals[0]/ev.sum()),
        "top8_fraction":float(vals[:8].sum()/ev.sum()),
        "effective_rank":float(ev.sum().square()/ev.square().sum()),
        "top_direction_heads":torch.argsort(shares[0],descending=True)[:3].tolist(),
        "cross_over_self_top1":float(cproj[0]/vals[0]),
    })
print("Spectral analysis seconds:",round(time.time()-t0,1))
print("Layer, top1%, top8%, effective rank, top direction heads, instruction/self energy")
for x in spectral_summary:
    print(x["layer"],round(100*x["top1_fraction"],2),round(100*x["top8_fraction"],2),
          round(x["effective_rank"],1),x["top_direction_heads"],round(x["cross_over_self_top1"],2))
np.savez(W/"action_output_spectral_reference.npz",
         **{f"L{l}_{k}":a for l,ss in action_spectra.items() for k,a in ss.items()})
del o,c,f,ev,u,U
note("- Added action output spectral reference (self-attention, cross-attention and FFN overlap). "
     "No causal or background-specific claim is made from weight-only spectra.")
```
