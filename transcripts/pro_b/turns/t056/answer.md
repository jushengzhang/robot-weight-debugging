Reducing BOS attention did not improve success, so I will revert that edit. I will now test a more specific routing intervention: an additive query shift for the noun-reading heads, chosen algebraically to favor non-BOS language while preserving the expert’s keys and values. The shift will also be orthogonal to pooled camera and BOS directions, rather than simply increasing all language attention.
```python
for n in list(edited()): drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 17-18: BOS sink localized, repair rejected\n"
            "Halving k_proj columns738,604 substantially reduces BOS attention "
            "(tomato L3 .263->.102,L6 .385->.207) but small action changes. "
            "Rollout60/128->56/128 (-3.1 +/-2.5pp), task20->18; "
            "semantic15->13. Sink attenuation alone is not beneficial; reverted.\n")
# Estimate the shared boundary component from the analytically known 1/n pooling factor.
# This is descriptive unmixing of activation summaries, not fitting model outputs.
invn=torch.tensor(1.0/lens,dtype=torch.float64)
ivc=invn-invn.mean()
freq=10000.0**(-torch.arange(0,256,2,dtype=torch.float64)/256)
def rope_vec(v, delta):
    c=torch.cos(freq*delta); s=torch.sin(freq*delta)
    a,b=v[:128],v[128:]
    return torch.cat([a*c-b*s,b*c+a*s])
qshift={}
qshift_info={}
for l,h in [(6,2),(12,2)]:
    gnorm=(1+load_original(f"{lm}.layers.{l}.input_layernorm.weight")).double()
    wk=load_original(f"{lm}.layers.{l}.self_attn.k_proj.weight").double()
    X={}
    for m in ["lang","base","wrist"]:
        X[m]=torch.tensor(np.stack([a[f"mean::vlm.{l-1}.resid::{m}"] for a in act3[:30]]),
                          dtype=torch.float64)
    boundary_slope=(ivc[:,None]*(X["lang"]-X["lang"].mean(0))).sum(0)/ivc.square().sum()
    content=X["lang"].mean(0)-invn.mean()*boundary_slope
    bos=content+boundary_slope
    def to_key(x):
        return (x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()*gnorm)@wk.T
    kt=rope_vec(to_key(content),-float(lens.mean()/2+5))
    kb=rope_vec(to_key(bos),-float(lens.mean()+5))
    camkeys=[]
    for m,pos in [("base",127.5),("wrist",383.5)]:
        K=to_key(X[m])
        camkeys.append(rope_vec(K.mean(0), pos-(512+float(lens.mean())+4)))
        _,_,vh=torch.linalg.svd(K-K.mean(0),full_matrices=False)
        for j in range(2):
            camkeys.append(rope_vec(vh[j],pos-(512+float(lens.mean())+4)))
    Q=load_original(f"{ex}.layers.{l}.self_attn.q_proj.weight").double().reshape(8,256,1024)
    Kex=load_original(f"{ex}.layers.{l}.self_attn.k_proj.weight").double()
    Vex=load_original(f"{ex}.layers.{l}.self_attn.v_proj.weight").double()
    constraints=[Kex,Vex]
    for hh in range(8):
        for kv in [kb]+camkeys:
            constraints.append((kv@Q[hh]/16)[None,:])
        if hh!=h:
            constraints.append((kt@Q[hh]/16)[None,:])
    # Keep the target-head query shift in slowly rotating RoPE coordinates.
    high=list(range(96))+list(range(128,224))
    constraints.append(Q[h,high,:])
    C=torch.cat(constraints,dim=0)
    C=C/C.norm(dim=1,keepdim=True).clamp_min(1e-12)
    _,s,vh=torch.linalg.svd(C,full_matrices=False)
    Vr=vh[s>s[0]*1e-8]
    grad=kt@Q[h]/16
    direction=grad-(grad@Vr.T)@Vr
    shift=direction/(direction@grad)
    qshift[l]=shift.float()
    qshift_info[l]={"target_head":h,"shift_rms_per_logit":float(shift.square().mean().sqrt()),
                    "kv_error":float(torch.cat([Kex,Vex])@shift @ (torch.cat([Kex,Vex])@shift)),
                    "null_rank":int(Vr.shape[0])}
    print("L",l,qshift_info[l],
          "actual target logit",round(float(grad@shift),4),
          "query shift norms",[round(float((q@shift).norm()),2) for q in Q])
torch.save({"shifts":qshift,"info":qshift_info},W/"subspaces/query_content_shift.pt")
```
