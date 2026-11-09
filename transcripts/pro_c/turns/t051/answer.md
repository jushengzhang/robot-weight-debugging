I will measure how much each MLP varies with scene identity versus the changed instruction. For the VLM, I can also compare instruction contrast before and after the MLP using the saved residuals. These are diagnostic correlations; any candidate still has to pass a paired behavioral test.
```python
ix17=json.loads((W/"probe_17"/"results.json").read_text())
a14=[np.load(r["activations"]) for r in ix14]
a17=[np.load(r["activations"]) for r in ix17]
def mat(arrs,key):
    return torch.tensor(np.stack([a[key] for a in arrs])).float()
def pair_stats(x):
    shared=(x[::2]+x[1::2])/2
    diff=x[1::2]-x[::2]
    var=shared-shared.mean(0)
    return float(var.norm(dim=-1).median()), float(diff.norm(dim=-1).median()), diff, shared
print("VLM layer: base scene/contrast | language contrast before->after MLP, MLP delta cosine")
for l in range(18):
    m=mat(a17,f"mean::vlm.{l}.mlp_out::lang")
    r=mat(a14,f"mean::vlm.{l}.resid::lang")
    mr=mat(a17,f"mean::vlm.{l}.mlp_out::base")
    sv,dv,_,_=pair_stats(mr)
    _,dm,md,_=pair_stats(m)
    _,dr,rd,_=pair_stats(r)
    pd=rd-md
    ratio=(rd.norm(dim=-1)/pd.norm(dim=-1)).median()
    cosine=torch.nn.functional.cosine_similarity(pd,md,dim=-1).median()
    print(l,round(sv,1),round(dv,1),round(float(pd.norm(dim=-1).median()),1),
          round(dr,1), "ratio",round(float(ratio),3),"cos",round(float(cosine),3))
print("EXPERT layer: MLP initial scene / instruction contrast / residual contrast, norm, top4 scene energy")
for l in range(18):
    m=mat(a17,f"calls::expert.{l}.mlp_out::action")[:,0,:]
    r=mat(a14,f"calls::expert.{l}.resid::action")[:,0,:]
    sv,dv,_,shared=pair_stats(m)
    _,dr,_,_=pair_stats(r)
    ev=torch.linalg.svdvals(shared-shared.mean(0))
    print(l,round(sv,2),round(dv,2),round(dr,2),
          round(float(m.norm(dim=-1).median()),2),round(float(ev[:4].square().sum()/ev.square().sum()),3))
# Persist compact metadata and all-development episode list.
(W/"dev_episodes.json").write_text(json.dumps(dev_eps))
print("EDITED",edited())
```
