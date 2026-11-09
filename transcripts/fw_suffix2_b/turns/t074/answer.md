The combined repair gains 4.2 points on development, but the paired uncertainty is still 3.9 points. Before freezing a candidate, I will check one more targeted distinction: whether the projection is spending capacity on suffix features that the gate already suppresses, instead of on the remaining corrupted task-token features.
```python
X=torch.from_numpy(np.load(W/"context_features_72x30x4096.npy")).double()
mask_idx=[1]+list(range(16,30))
Dall=proj["D"].double().reshape(24,30,4096)
Dm=Dall[:,mask_idx].reshape(-1,4096)
Q=proj["Qprotect"].double()
Dp=Dm-(Dm@Q)@Q.T
lm,sm,vm=torch.linalg.svd(Dp,full_matrices=False)
bm=Dm.T@lm/sm[None,:]
energy=torch.cumsum(sm**2,0)/torch.sum(sm**2)
print("MASKED-PREFIX SPECTRUM", {i:round(float(sm[i]),5) for i in [0,15,31,63,127,191,255,319,359]})
print("MASKED-PREFIX ENERGY RANKS",
      {f:int(torch.searchsorted(energy,torch.tensor(f))+1) for f in [.8,.9,.95,.99]})
for r in [64,128,192,256]:
    u=proj["U"][:,:r].double();b=proj["B_nearest"][:,:r].double()
    remaining=Dm-(Dm@u)@b.T
    print("ALL-CONTRAST",r,"prefix residual fraction",round(float(torch.linalg.norm(remaining)/torch.linalg.norm(Dm)),4))
    upr=vm[:r].T;bpr=bm[:,:r]
    rem=Dm-(Dm@upr)@bpr.T
    print("PREFIX-ONLY",r,"prefix residual fraction",round(float(torch.linalg.norm(rem)/torch.linalg.norm(Dm)),4),
          "basis norm",round(float(torch.linalg.norm(bpr)/np.sqrt(r)),4))
prefix_proj={"U":vm.T.float(),"B_nearest":bm.float(),"S":sm.float(),
             "Qprotect":Q.float(),"D":Dm.float(),"protect_rank":512}
torch.save(prefix_proj,W/"masked_prefix_projection.pt")
print("DEVELOPMENT HYBRID",json.dumps(paired_rolls([26])))
with (W/"notes.md").open("a") as f:
    f.write("\nPass26 full hybrid development96 raw: original48/96, edited52/96; +4.17pp, SE3.89, 9gains/5losses. Improvement is promising but NOT significant. Camera+12.5pp, Noise+18.75pp, Initial+6.25pp; Background/Light-6.25pp each. Checking whether gate-complementary masked-prefix projection gives more focused correction before final freeze.\n")
```
