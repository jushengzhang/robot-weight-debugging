Strengthening the late finger head did not advance reopening on the broader panel, so I will not claim a useful dose-response from that route. I will revert it and use the recorded layer-by-layer residual increments and MLP outputs to locate where the gripper decision is actually formed, rather than continuing to infer function from attention maps alone.
```python
clear_edits()
a4=np.stack([np.load(p["action"]) for p in p4])
g_targets=a4[:,:,6].mean(1)
width4=np.array([np.abs(torch.load(p["input"],map_location="cpu",weights_only=False)["element"]["observation/state"][-2:]).sum() for p in p4])
def corr(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else 0.
print("Layer gripper-axis projections: residual correlation with final grip, incremental correlation, MLP correlation; norm of MLP")
prev=None
for l in range(18):
    R=np.stack([x[f"calls::expert.{l}.resid::action"] for x in res4])
    M=np.stack([x[f"calls::expert.{l}.mlp_out::action"] for x in mlp13_orig])
    rg=R.mean(1)@gdir.numpy(); mg=M.mean(1)@gdir.numpy()
    dg=(R-prev).mean(1)@gdir.numpy() if prev is not None else rg
    print(l,round(corr(rg,g_targets),3),round(corr(dg,g_targets),3),round(corr(mg,g_targets),3),
          "mlp_norm",round(float(np.linalg.norm(M.mean(1),axis=1).mean()),2),
          "delta_g_sd",round(float(dg.std()),3))
    prev=R
print("Action output row cosines (first 7) to gripper:",ascii_round((AW[:7]@gdir)/(AW[:7].norm(dim=1)+1e-9),3))
print("Top gripper-readout channels by absolute weight:",[(int(j),round(float(AW[6,j]),4)) for j in torch.argsort(AW[6].abs(),descending=True)[:12]])
# Inspect per-layer MLP outputs' gripper-related change, not fit any model.
mlp_readout=[]
for l in range(18):
    M=np.stack([x[f"mean::expert.{l}.mlp_out::action"] for x in mlp13_orig])
    gp=M@gdir.numpy()
    mlp_readout.append((l,float(np.std(gp)),corr(gp,g_targets),corr(gp,width4)))
print("MLP axis variability (layer, std, grip correlation, width correlation):")
for z in sorted(mlp_readout,key=lambda x:-x[1])[:10]:
    print(z[0],*(round(v,3) for v in z[1:]))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 14: L15 H2 gain 2.0 not supported\n"
            "- 17 paired inputs, including 8 empty-closed loops and 8 holding-like successes: "
            "no gripper sign changes, small pose changes. Original threshold case also unchanged.\n"
            "- Reverted; no rollout claim. Continuing localization using actual residual increments and MLP outputs.\n")
```
