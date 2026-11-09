Softening E8H7 also failed to improve recovery: 31/64 became 30/64, with unchanged loop failures. I will revert it. For one final state-specific mechanism, I will check whether the final residual contains an empty-hand cue that the existing action readout does not use. If it does, a targeted gripper-readout edit could encourage reopening after a genuinely empty closure without changing global action scaling.
```python
drop_edit(qname)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 42: E8H7 query softening rejected\n"
            "- Gain 0.5: 31/64 -> 30/64, -1.6 pp (SE 2.7), 1 rescue / 2 losses, loops unchanged at 21. Both spatial-temperature doses lack a recovery gain. Reverted.\n")
X17=torch.stack([torch.from_numpy(np.load(r["activations"])["calls::expert.17.resid::action"]).float() for r in rr26])
fs,fh,fg=tm["final_modulation"].float().chunk(3,-1)
Yfin=X17/(X17.square().mean(-1,keepdim=True)+1e-6).sqrt()*(1+fs[None])+fh[None]
# Protect all existing action readout rows, not just the pose rows.
Wo=load_original("action_out_proj.weight")
protect=[Wo.T]
for gg in ["holding_like","open"]:
    for grip in ["open","close"]:
        ii=(groups==gg)&derive&np.array([p["grip"]==grip for p in phys])
        if ii.sum(): protect.append(Yfin[ii].mean(0).T)
P=torch.cat(protect,1)
uf,sf,_=torch.linalg.svd(P,full_matrices=False)
Pf=uf[:,sf>sf.max()*1e-5]
rawf=Yfin[(groups=="fully_closed")&derive].mean((0,1))-Yfin[(groups=="holding_like")&derive].mean((0,1))
df=rawf-Pf@(Pf.T@rawf); df/=df.norm()
sf1=(Yfin@df).numpy()
famp=float(sf1[(groups=="fully_closed")&derive].mean())
sfavg=sf1.mean(1)/famp
print("Final cue check means:",
      [(gg,round(float(sfavg[(groups==gg)&~derive].mean()),3),
        round(float(sfavg[(groups==gg)&~derive].std()),3)) for gg in ["fully_closed","holding_like","open"]])
ce=sfavg[(groups=="fully_closed")&~derive]
co=sfavg[(groups!="fully_closed")&~derive]
print("Scene-disjoint cue separation:",float((ce[:,None]>co).mean()))
print("Orthogonality to existing output rows (max cosine):",
      float(((Wo@df)/(Wo.norm(dim=1)+1e-12)).abs().max()))
print("Cue activation scale:",famp,"proposed row relative change beta=.25:",float((0.25/famp)/Wo[6].norm()))
```
