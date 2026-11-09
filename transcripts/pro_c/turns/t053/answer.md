Most of the late visual variation lies along a nearly common direction, rather than a clean scene-specific subspace. I will distinguish those possibilities before editing: suppressing a common visual offset is a different intervention from removing scene identity, and I will not describe one as the other.
```python
print("VLM visual common direction remaining after language protection")
for l in [13,14,15,16]:
    lang=mat(a17,f"mean::vlm.{l}.mlp_out::lang")
    vis=torch.cat([mat(a17,f"mean::vlm.{l}.mlp_out::base"),
                   mat(a17,f"mean::vlm.{l}.mlp_out::wrist")])
    q=ortho_columns(lang.T)
    mu=vis.mean(0); rm=mu-q@(q.T@mu)
    print(l,"visual mean norm",round(float(mu.norm()),1),
          "remaining",round(float(rm.norm()),1),
          "fraction",round(float(rm.norm()/mu.norm()),3))
print("EXPERT scene directions after instruction/mean protection; raw MLP outputs, first flow step")
expert_scene={}
for l in [4,6,8,9,10,11,12,13,14,15,16,17]:
    m=mat(a17,f"calls::expert.{l}.mlp_out::action")[:,0,:]
    scene=(m[::2]+m[1::2])/2
    contrasts=m[1::2]-m[::2]
    q=ortho_columns(torch.cat([contrasts.T, scene.mean(0)[:,None]],1))
    z=scene-scene.mean(0)
    zp=z-(z@q)@q.T
    _,s,vh=torch.linalg.svd(zp,full_matrices=False)
    u=vh[:4].T.contiguous()
    expert_scene[l]=u
    print(l,"retained",round(float(zp.norm()/z.norm()),3),
          "rank4",round(float(s[:4].square().sum()/s.square().sum()),3),
          "predicted MLP reduction",round(float((m@u).norm()/m.norm()),3))
torch.save(expert_scene,W/"expert_scene_mlp_bases.pt")
```
