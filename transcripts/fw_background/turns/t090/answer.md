I’ll now measure texture-sensitive residual directions after removing the light-control means and leading task-variation directions. I’ll also check agreement between two disjoint halves of the matched pairs; unstable directions will not be used for an edit.
```python
p19 = probe_results(19)
print("Probe record fields:", list(p19[0]))
actkey = next(k for k in p19[0] if "activ" in k)
acts19 = [dict(np.load(r[actkey])) for r in p19]
print("Residual key examples:", list(acts19[0])[:4], list(acts19[0])[-8:])

def get_resid(expert, layer):
    keys = [k for k in acts19[0] if k.startswith(f"mean::{expert}.{layer}.resid::")]
    keys.sort()
    return np.stack([[a[k] for k in keys] for a in acts19]), keys

npair = 54
fixed_data, fixed_stats = {}, {}
for L in range(1,30):
    X, keys = get_resid("video", L-1)
    assert X.shape[0] == 91 and X.shape[-1] == 3072, X.shape
    # Normalize camera-mean vectors as a proxy for the self-attention input basis.
    X = torch.from_numpy(X).float()
    X = X - X.mean(dim=-1, keepdim=True)
    X = X / X.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(1e-6)
    light = X[npair:]
    mean = light.mean(0)
    lc = (light - mean).reshape(-1,3072)
    _, ss, vv = torch.linalg.svd(lc, full_matrices=False)
    protect = torch.cat([mean.T, torch.ones(3072,1), vv[:8].T],1)
    Q = torch.linalg.qr(protect,mode="reduced").Q
    D = (X[:npair:2] - X[1:npair:2]).reshape(-1,3072)
    Dp = D - (D@Q)@Q.T
    _, ds, dv = torch.linalg.svd(Dp, full_matrices=False)
    U = dv[:2].T
    h0 = (X[:26:2]-X[1:26:2]).reshape(-1,3072)
    h1 = (X[26:54:2]-X[27:54:2]).reshape(-1,3072)
    h0 = h0 - (h0@Q)@Q.T
    h1 = h1 - (h1@Q)@Q.T
    u0 = torch.linalg.svd(h0,full_matrices=False).Vh[:2].T
    u1 = torch.linalg.svd(h1,full_matrices=False).Vh[:2].T
    ratios = ((D@U).square().mean(0)/(lc@U).square().mean(0).clamp_min(1e-12)).tolist()
    fixed_stats[L] = {
        "ratio":ratios,
        "contrast_fraction":float((D@U[:,:1]).square().sum()/D.square().sum()),
        "light_fraction":float((lc@U[:,:1]).square().sum()/lc.square().sum()),
        "leading_half_cosine":float(torch.abs(u0[:,0]@u1[:,0])),
        "old_direction_cosine":float(torch.abs(U[:,0]@torch.from_numpy(vd[f"U{L}"][:,0]))),
        "protected_light_fraction":float((ss[:8]**2).sum()/(ss**2).sum())
    }
    fixed_data[f"U{L}"] = U.numpy()
    fixed_data[f"Q{L}"] = Q.numpy()
    fixed_data[f"mean{L}"] = mean.numpy()
    fixed_data[f"X{L}"] = X.numpy()

print("\nL ratio1 contrast_fraction light_fraction split_cos old_cos protected_light")
for L, s in fixed_stats.items():
    print(L, *[round(float(v),4) for v in [
        s["ratio"][0],s["contrast_fraction"],s["light_fraction"],
        s["leading_half_cosine"],s["old_direction_cosine"],s["protected_light_fraction"]]])

np.savez_compressed(W / "fixed_video_residual_subspaces.npz", **fixed_data)
(W / "fixed_video_residual_stats.json").write_text(json.dumps(fixed_stats,indent=2))
print("\nNo edits:", edited())
```
