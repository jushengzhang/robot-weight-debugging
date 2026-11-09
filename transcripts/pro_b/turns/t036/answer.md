I’ll locate any layers where an MLP update contracts the matched instruction contrast, and distinguish that from ordinary residual growth. I’ll also compare instruction sensitivity with scene sensitivity in the expert MLPs; this may reveal a scene-prior pathway that attention-only edits missed.

```python
pr13=json.loads((W/"probe_13"/"results.json").read_text())
p13acts=[{k:v.copy() for k,v in np.load(workpath(r["activations"])).items()
          if k.startswith("mean::") or k.startswith("calls::")} for r in pr13]
def stack_act(acts,branch,l,what,group,inds=None,prefix="mean"):
    if inds is None: inds=range(len(acts))
    return torch.from_numpy(np.stack([acts[i][f"{prefix}::{branch}.{l}.{what}::{group}"] for i in inds])).float()
def pair_delta(x): return x[:30:2]-x[1:30:2]
print("VLM MLP instruction-contrast effects: layer after/pre norm, dot(pre,update)/pre2, update norm")
mlp_contraction=[]
for l in range(18):
    r=stack_act(p3acts,"vlm",l,"resid","lang")
    m=stack_act(p13acts,"vlm",l,"mlp_out","lang")
    da=pair_delta(r); dm=pair_delta(m); dp=da-dm
    ratio=float(da.norm()/dp.norm())
    dot=float((dp*dm).sum()/dp.square().sum())
    mn=float(dm.norm()/dp.norm())
    mlp_contraction.append((l,ratio,dot,mn))
    print(l,*[round(x,3) for x in (ratio,dot,mn)])
print("\nExpert MLP: layer task/scene contrast ratio, output RMS, task-delta RMS")
for l in range(18):
    m=stack_act(p13acts,"expert",l,"mlp_out","action")[:30]
    dt=pair_delta(m)
    ds=m[::2]-m[::2].roll(1,0)
    print(l,round(float(dt.norm()/ds.norm()),3),
          round(float(m.square().mean().sqrt()),3),
          round(float(dt.square().mean().sqrt()),3))
print("\nVLM visual MLP environment sensitivity delta/clean:")
for l in range(17):
    m=stack_act(p13acts,"vlm",l,"mlp_out","base")
    d=torch.stack([m[a]-m[b] for a,b in env_pairs])
    rr=stack_act(p3acts,"vlm",l,"resid","base")
    rd=torch.stack([rr[a]-rr[b] for a,b in env_pairs])
    print(l,round(float(d.norm()/rd.norm()),3))
```
