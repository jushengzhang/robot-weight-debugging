A single leading direction is not stable enough across task halves to justify an edit. I’ll test whether the halves share a small subspace instead, and protect enough light-control principal components to retain 95% of their variation. A readout edit is worth testing only if a reproducible texture direction survives that protection.
```python
Xraw = torch.from_numpy(get_resid("action",29)[0][:,0]).float()
light = Xraw[54:]
mu = light.mean(0)
lc = light-mu
_, s, vh = torch.linalg.svd(lc,full_matrices=False)
cum = torch.cumsum(s.square(),0)/s.square().sum()
protect_rank = int(torch.searchsorted(cum, torch.tensor(0.95)))+1
Q = torch.linalg.qr(torch.cat([mu[:,None],torch.ones(1024,1),vh[:protect_rank].T],1),
                   mode="reduced").Q
D = Xraw[:54:2]-Xraw[1:54:2]
Dp = D-(D@Q)@Q.T
B0 = torch.linalg.svd(Dp[:13],full_matrices=False).Vh[:4].T
B1 = torch.linalg.svd(Dp[13:],full_matrices=False).Vh[:4].T
left, overlap, right = torch.linalg.svd(B0.T@B1,full_matrices=False)
consensus = B0@left + B1@right.T
consensus = torch.linalg.qr(consensus,mode="reduced").Q

head_name = "mot.mixtures.action.head.weight"
H = load_original(head_name)
HQ = torch.linalg.qr(H.T,mode="reduced").Q
stats = []
for j in range(4):
    u = consensus[:,j]
    stats.append({
        "direction":j, "half_subspace_overlap":float(overlap[j]),
        "contrast_fraction":float((D@u).square().sum()/D.square().sum()),
        "light_fraction":float((lc@u).square().sum()/lc.square().sum()),
        "contrast_light_ratio":float((D@u).square().mean()/(lc@u).square().mean().clamp_min(1e-12)),
        "head_relative_rank1_norm":float((H@u).norm()/H.norm()),
        "head_rowspace_fraction":float((HQ.T@u).square().sum()),
        "protected_overlap":float((Q.T@u).norm())
    })
print("Light protected PCA rank:", protect_rank, "variance:", float(cum[protect_rank-1]))
print("Consensus directions:")
for r in stats:
    print({k:round(v,6) if isinstance(v,float) else v for k,v in r.items()})

np.savez_compressed(W / "readout_texture_subspace.npz",
    U=consensus.numpy(), Q=Q.numpy(), mean=mu.numpy(), D=D.numpy(), light_centered=lc.numpy())
(W / "readout_texture_subspace_stats.json").write_text(json.dumps(
    {"protect_rank":protect_rank,"directions":stats},indent=2))

# This is a predeclared mechanistic screen, not an outcome-driven fit.
eligible = [r["direction"] for r in stats
            if r["half_subspace_overlap"] >= 0.8
            and r["contrast_light_ratio"] >= 10
            and r["light_fraction"] <= 0.01]
print("Eligible direction indices:", eligible)
```
