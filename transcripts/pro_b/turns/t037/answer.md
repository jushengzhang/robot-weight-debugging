The VLM MLPs mostly preserve or expand instruction contrasts, which argues against a simple “language erased in the VLM” explanation. The strongest relative scene preference appears in expert MLPs 9–11. I’ll isolate their scene-varying output directions while protecting matched instruction differences and denoising-time means, then test whether suppressing that specific pathway changes task selection.

```python
scene_bases={}
scene_basis_stats=[]
for l in range(18):
    m=stack_act(p13acts,"expert",l,"mlp_out","action",inds=range(30),prefix="calls") # [30,10,1024]
    d=m[::2]-m[1::2]                          # matched instruction contrasts
    s=(m[::2]+m[1::2])/2                      # scene means
    tm=s.mean(0)                             # per-time mean, protect it
    dc=d.reshape(-1,1024)
    _,ds,dv=torch.linalg.svd(dc,full_matrices=False)
    # Protect all measured instruction contrast directions above numerical noise.
    rank_d=int((ds>ds.max()*1e-4).sum())
    protect_cols=torch.cat([dv[:rank_d].T,tm.T],1)
    q=torch.linalg.qr(protect_cols,mode="reduced").Q
    sc=(s-tm).reshape(-1,1024)
    clean=sc-(sc@q)@q.T
    _,ss,sv=torch.linalg.svd(clean,full_matrices=False)
    u=sv[:4].T.contiguous()
    scene_bases[l]=u
    frac=lambda x:float((x@u).square().sum()/(x.square().sum()+1e-12))
    scene_basis_stats.append([l,rank_d,round(frac(sc),3),round(frac(dc),7),
                              round(frac(m.reshape(-1,1024)),4)])
print("layer, protected contrast rank, scene-variance captured, instruction leakage, total MLP energy")
for x in scene_basis_stats: print(x)
torch.save(scene_bases,W/"subspaces"/"expert_scene_rank4.pt")
def scene_mlp_edit(beta=0.5,layers=(9,10,11),rank=4,reset=True):
    if reset: clear_edits()
    result=[]
    for l in layers:
        n=f"{EXP}.layers.{l}.mlp.down_proj.weight"
        w=load_original(n); u=scene_bases[l][:,:rank]
        delta=beta*u@(u.T@w)
        save_edit(n,w-delta)
        result.append((l,round(float(delta.norm()/w.norm()),4)))
    return result
print("candidate:",scene_mlp_edit())
snapshot("candidate_scene_mlp_half")
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 13: MLP diagnosis\n"
            "VLM MLP instruction contrast norm after/pre >=1 for all layers; no major cancellation. "
            "Expert MLPs9-11 have lowest task/scene contrast ratios0.628,0.651,0.655. "
            "Constructed rank4 scene-varying output bases orthogonal to all measured instruction "
            "differences and denoising-time means (initial observations, no success-label fitting).\n"
            "Candidate: left-project expert MLP down_proj9-11 with I-0.5 UU^T. "
            "Prediction: reduce scene-habit contributions while preserving instruction-dependent updates.\n")
```
