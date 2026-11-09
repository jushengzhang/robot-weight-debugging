The residual analysis shows that viewpoint differences accumulate across many blocks; there is no single dominant culprit. I will examine the final action readout for camera-sensitive directions outside the subspace used by the mild-light controls and action encoder. This could support a targeted readout edit without broadly rescaling actions.
```python
clean = torch.from_numpy(RG[:,2,29].reshape(-1,1024))
dc = clean - clean.mean(0)
uc, sc, vc = torch.linalg.svd(dc, full_matrices=False)
cume = sc.square().cumsum(0)/sc.square().sum()
ranks = {p:int(torch.searchsorted(cume,torch.tensor(p)).item()+1) for p in [.9,.95,.99,.995,.999]}
print("Clean centered ranks:",ranks)
# Protect 99% of clean-state variation, its mean, and the action-encoder column span.
rank_clean = ranks[.99]
encoder = load_original("mot.mixtures.action.action_encoder.weight")
protect_raw = torch.cat([vc[:rank_clean].T, clean.mean(0)[:,None], encoder],1)
protect_basis = torch.linalg.qr(protect_raw,mode="reduced").Q
dlast = torch.from_numpy(DR[:,29].mean(1))
dprotected = dlast - (dlast @ protect_basis) @ protect_basis.T
up, sp, vp = torch.linalg.svd(dprotected, full_matrices=False)
print("Protected subspace rank:",protect_basis.shape[1])
print("Camera contrast energy outside protected subspace:",round(float(dprotected.square().sum()/dlast.square().sum()),4))
print("Projected camera contrast top8 energies:",np.round((sp[:8].square()/sp.square().sum()).numpy(),4).tolist())
print("Head weight energy fractions in these camera directions:",
      np.round((head@vp[:8].T).square().sum(0).div(head.square().sum()).numpy(),5).tolist())
_,sa,va = torch.linalg.svd(dprotected[::2],full_matrices=False)
_,sb,vb = torch.linalg.svd(dprotected[1::2],full_matrices=False)
for k in [1,2,4,8]:
    held = (dprotected[1::2]@va[:k].T).square().sum()/dprotected[1::2].square().sum()
    align = torch.linalg.svdvals(va[:k]@vb[:k].T)
    print("rank",k,"held-half contrast energy",round(float(held),4),
          "subspace cosines",np.round(align.numpy(),3).tolist())
np.savez(W/"readout_protected_camera_subspace.npz",
         protect_basis=protect_basis.numpy(), camera_basis=vp.numpy(),
         camera_singular=sp.numpy(), clean_rank=np.array(rank_clean))
```
