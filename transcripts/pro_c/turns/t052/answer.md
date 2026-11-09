The VLM’s layer-16 MLP stands out: its camera-mean scene variation is much larger than in earlier layers, and its output feeds the final prefix KV cache. I will check whether this variation contains a low-rank visual component that can be attenuated while preserving the measured language outputs and shared residual directions.
```python
print("Late VLM MLP norms and dominant coordinates")
for l in [12,13,14,15,16,17]:
    for g in ["base","wrist","lang"]:
        x=mat(a17,f"mean::vlm.{l}.mlp_out::{g}")
        _,s,vh=torch.linalg.svd(x-x.mean(0),full_matrices=False)
        top=torch.topk(x.square().mean(0),5)
        print(l,g,"norm",round(float(x.norm(dim=-1).median()),1),
              "rank4",round(float(s[:4].square().sum()/s.square().sum()),3),
              "coords",[(int(i),round(float(z),1)) for i,z in zip(top.indices,top.values.sqrt())])
def ortho_columns(x,rtol=1e-5):
    u,s,_=torch.linalg.svd(x,full_matrices=False)
    return u[:,s>s[0]*rtol]
scene_bases={}
for l in [13,14,15,16]:
    lang=mat(a17,f"mean::vlm.{l}.mlp_out::lang")
    base=mat(a17,f"mean::vlm.{l}.mlp_out::base")
    wrist=mat(a17,f"mean::vlm.{l}.mlp_out::wrist")
    common=torch.cat([(base[::2]+base[1::2])/2,(wrist[::2]+wrist[1::2])/2])
    sinks=torch.eye(2048)[:,[50,604,674,738]]
    protect=torch.cat([lang.T, common.mean(0)[:,None], sinks],dim=1)
    q=ortho_columns(protect)
    scene=common-common.mean(0)
    sproj=scene-(scene@q)@q.T
    _,s,vh=torch.linalg.svd(sproj,full_matrices=False)
    u=vh[:4].T.contiguous()
    scene_bases[l]=u
    print("visual-scene candidate",l,"protected rank",q.shape[1],
          "scene retained",round(float(sproj.norm()/scene.norm()),3),
          "rank4 energy",round(float(s[:4].square().sum()/s.square().sum()),3),
          "lang leakage",round(float((lang@u).norm()/lang.norm()),7))
torch.save(scene_bases,W/"visual_scene_mlp_bases.pt")
```
