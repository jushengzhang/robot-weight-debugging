The probe provides token-group averages, not token-level residuals, so it cannot directly identify a BOS-specific activation direction. I’ll instead test a narrower, supportable hypothesis: instruction changes are represented in the VLM but underweighted when their values reach the expert. I’ll construct a low-rank instruction subspace from matched instruction contrasts, explicitly removing dominant visual directions.

```python
VLM="paligemma_with_expert.paligemma.model.language_model"
EXP="paligemma_with_expert.gemma_expert.model"
VIS="paligemma_with_expert.paligemma.model.vision_tower.vision_model"
p3acts=[{k:v.copy() for k,v in np.load(workpath(r["activations"])).items()
         if k.startswith("mean::") or k.startswith("calls::expert")} for r in pr3]
def stack3(layer,group,branch="vlm",indices=None):
    inds=range(len(p3acts)) if indices is None else indices
    return torch.from_numpy(np.stack([p3acts[i][f"mean::{branch}.{layer}.resid::{group}"] for i in inds]))
def unit_rows(x):
    return x/(x.norm(dim=-1,keepdim=True)+1e-9)
for l in (0,3,6,9,12,15,17):
    x=stack3(l,"lang")[:30]
    y=stack3(l,"base")[:30]
    intra_lang=(x[::2]-x[1::2]).norm(dim=1).mean()
    inter_lang=(x[::2]-x[::2].roll(1,0)).norm(dim=1).mean()
    msg=[l,round(float(intra_lang/inter_lang),3)]
    if l in (3,6,9,12,15,17):
        z=stack3(l,"action","expert")[:30]
        msg += [round(float((z[::2]-z[1::2]).norm(dim=1).mean() /
                            (z[::2]-z[::2].roll(1,0)).norm(dim=1).mean()),3)]
    else: msg += [None]
    print("layer/task-vs-scene distance ratio [VLM lang, expert]:",msg)
for l in (3,6,9,12):
    w=load_original(f"{VLM}.layers.{l}.input_layernorm.weight")
    print("ln",l, "min/median/max", [round(float(v),3) for v in (w.min(),w.median(),w.max())])

text_bases={}
basis_stats=[]
for l in range(1,18):
    ln=1+load_original(f"{VLM}.layers.{l}.input_layernorm.weight")
    # These are normalized pooled-feature proxies, not a local network forward.
    xb=unit_rows(stack3(l-1,"base"))*ln
    xw=unit_rows(stack3(l-1,"wrist"))*ln
    xt=unit_rows(stack3(l-1,"lang"))*ln
    visual=torch.cat([xb,xw],0)
    vm=visual.mean(0,keepdim=True)
    _,sv,vh=torch.linalg.svd(visual-vm,full_matrices=False)
    protect=torch.cat([xb.mean(0,keepdim=True),xw.mean(0,keepdim=True),
                       xt[:30].mean(0,keepdim=True),vh[:12]],0).T
    q=torch.linalg.qr(protect,mode="reduced").Q
    d=xt[:30:2]-xt[1:30:2]
    clean=d-(d@q)@q.T
    _,sd,vd=torch.linalg.svd(clean,full_matrices=False)
    u=vd[:8].T.contiguous()
    text_bases[l]=u
    wk=load_original(f"{VLM}.layers.{l}.self_attn.k_proj.weight")
    wv=load_original(f"{VLM}.layers.{l}.self_attn.v_proj.weight")
    frac=lambda x: float((x@u).square().sum()/x.square().sum())
    basis_stats.append([l,round(frac(xt),3),round(frac(visual),4),
                        round(float((wv@u).norm()/wv.norm()),3),
                        round(float((wk@u).norm()/wk.norm()),3),
                        round(float(sd[:8].square().sum()/sd.square().sum()),3)])
print("subspace stats: layer, text energy, image energy, V-rank norm, K-rank norm, contrast energy")
for row in basis_stats: print(row)
(W/"subspaces").mkdir(exist_ok=True)
torch.save(text_bases,W/"subspaces"/"instruction_contrast_rank8.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 2-3 diagnosis\n"
            "Task instructions remain distinct in VLM residuals. Expert attention in layers 3/6/9/12 "
            "is often dominated by BOS (roughly 13-38% averaged over heads); language mass alone "
            "is not content use. Several wrist heads fixate on the same patch across scenes "
            "(L3 H3, L9 H2). Probe residuals are group-averaged, not per-token. "
            "Constructed rank-8 VLM instruction-contrast subspaces at each layer, orthogonal "
            "to camera means, common text mean, and 12 leading centered visual PCs. "
            "Directions use matched initial observations, not success labels or optimization.\n")
```
