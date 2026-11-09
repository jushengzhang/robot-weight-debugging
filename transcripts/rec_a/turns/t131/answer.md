I will form a single empty-hand contrast from the recorded activations, not fit a predictor. Holding/open-hand means, denoising-time means, and direct action-input directions will be protected. A scene-disjoint half of these observations will check whether the contrast detects the physical state rather than memorizing particular scenes.
```python
import hashlib
E="paligemma_with_expert.gemma_expert.model."
tm=torch.load(W/"recovered_time_modulation.pt",weights_only=False)
C=tm["cond"].float()
groups=np.array([p["group"] for p in phys])
derive=np.array([int(hashlib.sha256(("empty-query:"+str(p["id"])).encode()).hexdigest()[:8],16)%2==0 for p in phys])
print("State counts derive/check:",[(g,int(((groups==g)&derive).sum()),int(((groups==g)&~derive).sum())) for g in sorted(set(groups))])
Xin=torch.stack([torch.from_numpy(np.load(r["activations"])["calls::expert.15.resid::action"]).float() for r in rr26])
mod=C@load_original(E+"layers.16.input_layernorm.dense.weight").T+load_original(E+"layers.16.input_layernorm.dense.bias")
scale,shift,gate=mod.chunk(3,dim=-1)
Xnorm=Xin/(Xin.square().mean(-1,keepdim=True)+1e-6).sqrt()
Xq=Xnorm*(1+scale[None])+shift[None]
Ain=load_original("action_in_proj.weight")[:,:7]
protect=[Ain]
for g in ["holding_like","open"]:
    ii=(groups==g)&derive
    protect.append(Xq[ii].mean(0).T)
for g0 in ["open","close"]:
    ii=np.array([p["grip"]==g0 for p in phys])&derive
    protect.append(Xq[ii].mean(0).T)
P=torch.cat(protect,dim=1)
Up,Sp,Vp=torch.linalg.svd(P,full_matrices=False)
Porth=Up[:,Sp>Sp.max()*1e-5]
raw=(Xq[(groups=="fully_closed")&derive].mean((0,1))
     -Xq[(groups=="holding_like")&derive].mean((0,1)))
d=raw-Porth@(Porth.T@raw)
d=d/d.norm()
scores=(Xq@d).numpy()
norm_scale=float(scores[(groups=="fully_closed")&derive].mean())
print("Contrast residual fraction:",round(float((raw-Porth@(Porth.T@raw)).norm()/raw.norm()),4))
for check,mask in [("derive",derive),("check",~derive)]:
    print(check,[(g,np.round(scores[(groups==g)&mask].mean(0)[[0,4,9]]/norm_scale,3).tolist(),
                  round(float(scores[(groups==g)&mask].std()/norm_scale),3)) for g in ["fully_closed","holding_like","open"]])
ss=scores.mean(1)/norm_scale
check_empty=ss[(groups=="fully_closed")&~derive]
check_other=ss[(groups!="fully_closed")&~derive]
print("Scene-disjoint pairwise separation:",float((check_empty[:,None]>check_other).mean()))
print("Principal-mode file:", {k:tuple(v.shape) if torch.is_tensor(v) else str(v)[:120]
                              for k,v in torch.load(W/"E16H3_top_query_mode.pt",weights_only=False).items()})
```
