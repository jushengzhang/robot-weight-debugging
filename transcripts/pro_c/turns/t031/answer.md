The early instruction-head edits did not establish a useful repair. I will now examine the shared visual shift in environment failures: if table/scene changes occupy a compact direction in the vision features, a reference-centered projection could suppress that nuisance while preserving the spatial and object-dependent residuals.
```python
def activation_matrix(records, key):
    return np.stack([np.load(r["activations"])[key] for r in records])
for l in [0,8,16,26]:
    ds=[]; refs=[]
    for suite,t,i in chosen_scenes:
        g=by_scene[(suite,t)][i]
        a=acts1[g["environment"]["id"]][f"mean::vision.{l}.resid::base"]
        b=acts1[g["semantic"]["id"]][f"mean::vision.{l}.resid::base"]
        ds.append(a-b); refs.append(b)
    D=np.stack(ds); R=np.stack(refs)
    u,s,vt=np.linalg.svd(D,full_matrices=False)
    ev=(s*s)/(s*s).sum()
    print("vision",l,"environment difference rank energy top1/2/4:",
          [round(float(ev[:k].sum()),3) for k in [1,2,4]],
          "mean-difference/RMS ref",round(float(np.linalg.norm(D.mean(0))/np.linalg.norm(R.mean(0))),3))
# Initial-query check, independent of closed-loop trajectory differences.
D0=np.stack([np.load(p3[j+3]["activations"])["mean::vision.26.resid::base"]-
             np.load(p3[j]["activations"])["mean::vision.26.resid::base"] for j in [0,4,8,12]])
u0,s0,vt0=np.linalg.svd(D0,full_matrices=False)
print("initial4 scene environment difference top2 energy:",np.round(s0[:2]**2/(s0**2).sum(),3).tolist())
print("initial environment contrast cosine matrix:",np.round(D0@D0.T/(np.linalg.norm(D0,axis=1)[:,None]*np.linalg.norm(D0,axis=1)[None,:]),2).tolist())
print("projector dimensions:",SHAPES["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"])
```
