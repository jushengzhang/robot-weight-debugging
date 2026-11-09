The environment shift is low-rank early in vision but becomes distributed by the final encoder layer, so I will not impose an arbitrary final-feature correction. I will test one remaining task-focused candidate: attenuating scene-shared components in expert MLPs 12–13, immediately before the large instruction-sensitive increase at layer 14. The construction will protect measured instruction contrasts and mean denoising-time responses across all ten flow steps.
```python
expert_scene_time={}
for l in [9,10,11,12,13]:
    m=mat(a17,f"calls::expert.{l}.mlp_out::action") # [32,10,1024]
    shared=(m[::2]+m[1::2])/2
    contrast=(m[1::2]-m[::2]).reshape(-1,1024)
    time_mean=shared.mean(0)
    q=ortho_columns(torch.cat([contrast,time_mean],0).T,rtol=1e-4)
    z=(shared-time_mean).reshape(-1,1024)
    zp=z-(z@q)@q.T
    _,s,vh=torch.linalg.svd(zp,full_matrices=False)
    u=vh[:4].T.contiguous()
    expert_scene_time[l]=u
    print("expert",l,"protected rank",q.shape[1],
          "scene remaining",round(float(zp.norm()/z.norm()),3),
          "rank4 energy",round(float(s[:4].square().sum()/s.square().sum()),3),
          "output fraction",round(float((m@u).norm()/m.norm()),3),
          "contrast leakage",round(float((contrast@u).norm()/contrast.norm()),7))
torch.save(expert_scene_time,W/"expert_scene_time_bases.pt")
for l in [12,13]:
    n=f"{EX}.layers.{l}.mlp.down_proj.weight"
    v=load_original(n); u=expert_scene_time[l]
    save_edit(n,v-0.5*u@(u.T@v))
print("H weight changes",[(n.split(".layers.")[1].split(".")[0],round(float((load(n)-load_original(n)).norm()/load_original(n).norm()),4)) for n in edited()])
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate H: expert scene-shared MLP components\nClosed-form rank4 scene covariance basis in expert MLP12/13 outputs, projected orthogonal to160 matched instruction contrasts (16pairs x10 denoising steps) and10 mean-time outputs. Halve these components via left projection of down_proj only. Tests whether scene-shared pre-L14 processing obstructs task contrast; no claim these directions are necessarily spurious before behavior test. Environment correction at final vision output not pursued: rank1 shift energy falls from~98% L0 to20% L26.\n")
```
