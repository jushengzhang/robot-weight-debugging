The first construction was degenerate: protecting the all-state means also removed the empty-state contrast algebraically. No weights were changed. I will correct the protection set to use only holding/open controls and their noise groups, then repeat the separation check.
```python
protect=[Ain]
for g in ["holding_like","open"]:
    for grip in ["open","close"]:
        ii=(groups==g)&derive&np.array([p["grip"]==grip for p in phys])
        if ii.sum():
            protect.append(Xq[ii].mean(0).T)
P=torch.cat(protect,dim=1)
Up,Sp,Vp=torch.linalg.svd(P,full_matrices=False)
Porth=Up[:,Sp>Sp.max()*1e-5]
d0=raw-Porth@(Porth.T@raw)
d=d0/d0.norm()
scores=(Xq@d).numpy()
norm_scale=float(scores[(groups=="fully_closed")&derive].mean())
print("Contrast residual fraction:",round(float(d0.norm()/raw.norm()),4),"raw score scale",round(norm_scale,4))
for check,mask in [("derive",derive),("check",~derive)]:
    print(check,[(g,np.round(scores[(groups==g)&mask].mean(0)[[0,4,9]]/norm_scale,3).tolist(),
                  round(float(scores[(groups==g)&mask].std()/norm_scale),3)) for g in ["fully_closed","holding_like","open"]])
ss=scores.mean(1)/norm_scale
check_empty=ss[(groups=="fully_closed")&~derive]
check_other=ss[(groups!="fully_closed")&~derive]
print("Scene-disjoint pairwise separation:",float((check_empty[:,None]>check_other).mean()))
print("Check scores by scene:",[(phys[i]["id"],phys[i]["group"],round(float(ss[i]),3)) for i in np.flatnonzero(~derive)])
```
