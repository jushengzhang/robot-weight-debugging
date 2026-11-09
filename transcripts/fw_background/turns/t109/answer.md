The failure-specific MLP mean shifts are not consistent enough across tasks. I’ll check whether the tasks instead share a variance direction with opposite signs—for example, different textures pushing control errors in opposite directions. Only a shared subspace with low variation on successful controls will be considered for the causal test.
```python
mlp_consensus_data,mlp_consensus_stats = {},[]
for L in range(30):
    D = torch.from_numpy(mlp_data[f"D{L}"])
    QL = torch.from_numpy(mlp_data[f"Q{L}"])
    DP = D-(D@QL)@QL.T
    TD = torch.stack([DP[ix].mean(0) for ix in group_indices])
    B0 = torch.linalg.svd(TD[::2],full_matrices=False).Vh.T
    B1 = torch.linalg.svd(TD[1::2],full_matrices=False).Vh.T
    a,ov,b = torch.linalg.svd(B0.T@B1,full_matrices=False)
    C = torch.linalg.qr(B0@a+B1@b.T,mode="reduced").Q
    controls = torch.from_numpy(mlp_data[f"C{L}"])
    controls = controls-controls.mean(0)
    for j in range(min(2,C.shape[1])):
        direction = C[:,j]
        fv = float((D@direction).square().mean())
        cv = float((controls@direction).square().mean())
        frac = float((D@direction).square().sum()/D.square().sum())
        r = {"layer":L,"direction":j,"overlap":float(ov[j]),
             "failure_control_ratio":fv/max(cv,1e-12),
             "failure_fraction":frac,
             "failure_rms":math.sqrt(fv),"control_rms":math.sqrt(cv)}
        mlp_consensus_stats.append(r)
    mlp_consensus_data[f"U{L}"] = C.numpy()
    mlp_consensus_data[f"Q{L}"] = QL.numpy()
    mlp_consensus_data[f"mean{L}"] = mlp_data[f"mean{L}"]

print("Highest cross-task shared-subspace overlaps:")
for r in sorted(mlp_consensus_stats,key=lambda r:r["overlap"],reverse=True)[:18]:
    print({k:round(v,5) if isinstance(v,float) else v for k,v in r.items()})
eligible_mlp = [r for r in mlp_consensus_stats
                if r["overlap"] >= .8 and r["failure_control_ratio"] >= 10]
print("Eligible:",[(r["layer"],r["direction"]) for r in eligible_mlp])
np.savez_compressed(W/"failure_mlp_consensus.npz",**mlp_consensus_data)
(W/"failure_mlp_consensus_stats.json").write_text(json.dumps(mlp_consensus_stats,indent=2))
```
