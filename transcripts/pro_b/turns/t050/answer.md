Doubling the instruction-reading heads did not help: task success fell from 20/68 to 17/68, and old-task completions stayed at 35. I will revert it. The next question is whether attention routing—not the strength of the retrieved values—is the bottleneck. I will inspect modality-separating key directions and residual outliers before attempting a routing-specific edit.
```python
for n in list(edited()): drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 16: instruction-head output gain rejected\n"
            "Expert L12H2,L15H0 o_proj blocks x2:60/128->58/128 "
            "(-1.6 +/-2.2pp); task20->17/68, old-goal failures35 unchanged; "
            "object14->12. Reverted. Strengthening retrieved content is insufficient.\n")
# Residual summaries: outlier dimensions and pooled modality separation.
act3=[dict(np.load(x["activations"])) for x in pr[3]]
for l in [0,1,2,3,5,6,8,9,11,12,14,15,17]:
    arrays={m:np.stack([a[f"mean::vlm.{l}.resid::{m}"] for a in act3[:30]]) 
            for m in ["base","wrist","lang"]}
    lines=[]
    for m,X in arrays.items():
        v=np.abs(X).mean(0)
        ix=np.argsort(v)[-4:][::-1]
        lines.append((m,round(float(np.linalg.norm(X,axis=1).mean()),1),
                      [(int(i),round(float(v[i]),2)) for i in ix],
                      round(float(np.median(v)),3)))
    print("L",l,lines)
# Adaptive normalization chunks; no forward computation.
print("\nExpert adaptive input-norm bias chunks: scale, shift, gate")
for l in [0,3,6,9,12,15,17]:
    b=load_original(f"{ex}.layers.{l}.input_layernorm.dense.bias").reshape(3,1024)
    print(l,"RMS",[round(float(t.square().mean().sqrt()),3) for t in b],
          "range",[(round(float(t.min()),2),round(float(t.max()),2)) for t in b])
print("\nPooled VLM key proxies, text-vs-image separation:")
key_stats={}
for l in [3,6,9,12,15]:
    gamma=1+load_original(f"{lm}.layers.{l}.input_layernorm.weight")
    wk=load_original(f"{lm}.layers.{l}.self_attn.k_proj.weight")
    keymeans={}
    for m in ["base","wrist","lang"]:
        x=torch.tensor(np.stack([a[f"mean::vlm.{l-1}.resid::{m}"] for a in act3[:30]]))
        xn=x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()*gamma
        keymeans[m]=xn@wk.T
    diff=keymeans["lang"].mean(0)-(keymeans["base"].mean(0)+keymeans["wrist"].mean(0))/2
    key_stats[l]=(keymeans,diff)
    print(l, "modality_diff norm",round(float(diff.norm()),2),
          "meanK norms",{m:round(float(k.mean(0).norm()),1) for m,k in keymeans.items()},
          "low-frequency fraction",round(float(diff.reshape(2,128)[:,96:].square().sum()/diff.square().sum()),3))
```
