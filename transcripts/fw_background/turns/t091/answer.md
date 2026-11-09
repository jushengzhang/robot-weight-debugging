The probe archive appears to contain only action residuals, so I cannot use it to confirm a fixed-input video subspace. I’ll check the schema and instead use the recorded action residuals to locate texture differences that actually reach the policy’s readout, with light-task variation explicitly protected.
```python
print("Activation key count:", len(acts19[0]))
print("Non-action keys:", [k for k in acts19[0] if "action." not in k][:20])
print("\nFinal action projection/modulation names:")
for n in NAMES:
    if "action" in n and ".blocks." not in n:
        print(n, SHAPES[n])

action_fixed_data, action_fixed_stats = {}, {}
for L in range(30):
    X, keys = get_resid("action", L)
    assert X.shape == (91,1,1024), (L,X.shape,keys)
    X = torch.from_numpy(X[:,0]).float()
    X = X-X.mean(-1,keepdim=True)
    X = X/X.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
    light = X[54:]
    mu = light.mean(0)
    lc = light-mu
    _, ss, vv = torch.linalg.svd(lc,full_matrices=False)
    Q = torch.linalg.qr(torch.cat([mu[:,None],torch.ones(1024,1),vv[:8].T],1),
                       mode="reduced").Q
    D = X[:54:2]-X[1:54:2]
    Dp = D-(D@Q)@Q.T
    _, ds, dv = torch.linalg.svd(Dp,full_matrices=False)
    U = dv[:2].T
    halves = []
    for dh in [D[:13], D[13:]]:
        dh = dh-(dh@Q)@Q.T
        halves.append(torch.linalg.svd(dh,full_matrices=False).Vh[0])
    action_fixed_stats[L] = {
        "ratio":((D@U).square().mean(0)/(lc@U).square().mean(0).clamp_min(1e-12)).tolist(),
        "contrast_fraction":float((D@U[:,:1]).square().sum()/D.square().sum()),
        "light_fraction":float((lc@U[:,:1]).square().sum()/lc.square().sum()),
        "leading_half_cosine":float(torch.abs(halves[0]@halves[1])),
        "protected_light_fraction":float((ss[:8]**2).sum()/(ss**2).sum()),
        "residual_pair_rms":float(D.square().mean().sqrt())
    }
    action_fixed_data[f"U{L}"] = U.numpy()
    action_fixed_data[f"Q{L}"] = Q.numpy()
    action_fixed_data[f"mean{L}"] = mu.numpy()
    action_fixed_data[f"X{L}"] = X.numpy()

print("\nL ratio1 contrast_frac light_frac split_cos protected_light pair_RMS")
for L,s in action_fixed_stats.items():
    print(L,*[round(float(v),4) for v in [
        s["ratio"][0],s["contrast_fraction"],s["light_fraction"],
        s["leading_half_cosine"],s["protected_light_fraction"],s["residual_pair_rms"]]])

np.savez_compressed(W / "fixed_action_residual_subspaces.npz",**action_fixed_data)
(W / "fixed_action_residual_stats.json").write_text(json.dumps(action_fixed_stats,indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\n## Pass19: fixed-input residual localization\n")
    f.write("- 54 paired initial inputs + 37 light controls, original model. Archive contains only action residuals; fixed-input VIDEO residual hypothesis cannot be confirmed from this tool output.\n")
    f.write("- Saved task-protected action residual texture bases, with split-half stability and light-variance leakage. No new edits yet.\n")
```
