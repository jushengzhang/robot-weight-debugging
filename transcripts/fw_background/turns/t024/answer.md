The projection submatrices are not simple copies, so I will not assume inherited feature coordinates. I will compare head-space covariance geometry instead. This tests a narrower structural question: whether the two experts use compatible directions within each shared attention head despite their different hidden widths.

```python
cov_geometry=[]
for l in range(30):
    rec={"layer":l}
    for proj in ["q","k","v"]:
        a=load_original(pname("action",l,f"self_attn.{proj}.weight")).reshape(24,128,1024)
        v=load_original(pname("video",l,f"self_attn.{proj}.weight")).reshape(24,128,3072)
        ga=torch.bmm(a,a.transpose(1,2))
        gv=torch.bmm(v,v.transpose(1,2))
        eye=torch.eye(128)[None,:,:]
        ga=ga-ga.diagonal(dim1=1,dim2=2).mean(1)[:,None,None]*eye
        gv=gv-gv.diagonal(dim1=1,dim2=2).mean(1)[:,None,None]*eye
        sim=(ga*gv).sum((1,2))/(ga.square().sum((1,2))*gv.square().sum((1,2))).sqrt().clamp_min(1e-12)
        rec[proj]=sim.numpy()
    cov_geometry.append(rec)
print("Layer: median covariance-geometry cosine [Q,K,V], V range")
for l in [0,1,3,6,9,12,15,18,21,24,27,29]:
    ss=cov_geometry[l]
    print(l,[round(float(np.median(ss[p])),3) for p in ["q","k","v"]],
          [round(float(ss["v"].min()),3),round(float(ss["v"].max()),3)])
np.savez(W/"expert_head_covariance_similarity.npz",
         **{f"L{x['layer']}_{p}":x[p] for x in cov_geometry for p in ["q","k","v"]})
del a,v,ga,gv
note("- Action/video projection slices have near-zero coordinate-wise cosine; do not assume a simple copied/truncated feature basis. "
     "Q/K normalization vectors remain similar, and head-space covariance comparisons are saved for reference.")
```
