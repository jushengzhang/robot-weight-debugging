The video readout is complete. I’ll save it, inspect how much independent structure the weighted contrasts contain, and take one matching readout of the smaller action-text MLP. That will let me test the two text-entry routes separately.
```python
z20=collect_weighted_readout(20)
weighted=torch.zeros(128,13,3072,dtype=torch.float64)
for sidx in range(3):
    ids=spec["feature_ids"][sidx*1018:(sidx+1)*1018]
    weighted[:,:,ids]=torch.from_numpy(np.load(W/f"weighted_video_{sidx}.npy"))
torch.save(weighted,W/"weighted_video_features.pt")
# Look at static spectra; scale each weighting equally in the SVD.
sc=weighted[48:].square().mean((0,2)).sqrt().clamp_min(1e-6)
X=(weighted[:48]-weighted[48:96])/sc[None,:,None]
Cw=weighted[48:]/sc[None,:,None]
for label,arr in [("weighted contrasts",X),("weighted clean",Cw)]:
    mat=arr.reshape(-1,3072).T
    _,sv,_=torch.linalg.svd(mat,full_matrices=False)
    energy=sv.square().cumsum(0)/sv.square().sum()
    print(label,"top10",sv[:10].round(decimals=4).tolist(),
          "ranks", {str(e):int(torch.searchsorted(energy,e))+1 for e in [.9,.99,.999,.9999]},
          "singular_at_64_128_256_512",[round(float(sv[min(i,len(sv)-1)]),5) for i in [63,127,255,511]])
install_weighted_readout(0,expert="action")
```
