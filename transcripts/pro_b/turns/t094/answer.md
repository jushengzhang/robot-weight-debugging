The controlled visual-value gain also failed, so I will revert it. Before another edit, I will check whether the instruction signal is actually cancelled later in the expert or merely diluted by growing residual norms; those cases require different interventions.
```python
for n in list(edited()):drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass28: controlled visual-value gain rejected\n"
            "VLM14 value rank8 gain2:66/128->64/128 (-1.6 +/-2.2pp), task13->12,position7->6; "
            "old-goal failures10->12. Object/semantic unchanged. Restored original. "
            "Even strengthening the causally isolated instruction-dependent visual span does not improve target selection.\n")
print("Expert controlled differences at first denoise step:")
prev=None
for l in [12,13,14,15,16,17]:
    X=get_stack("orig",f"calls::expert.{l}.resid::action")[:,0]
    Y=get_stack("edit",f"calls::expert.{l}.resid::action")[:,0]
    D=Y-X
    vals={"X_norm":float(X.norm(dim=1).median()),"D_norm":float(D.norm(dim=1).median())}
    if prev is not None:
        keep=prev.norm(dim=1)>0.01
        vals["delta_retained_parallel"]=float(((D*prev).sum(1)/(prev.square().sum(1)+1e-10))[keep].median())
        vals["delta_cos"]=float(torch.nn.functional.cosine_similarity(D[keep],prev[keep],dim=1).median())
    print(l,{k:round(v,3) for k,v in vals.items()})
    prev=D
for l in [0,3,6,9,12,14,15,16,17]:
    n=EP+f"layers.{l}.input_layernorm.dense.bias"
    b=load_original(n).reshape(3,1024)
    print("adaptive bias",l,"rms",b.square().mean(1).sqrt().round(decimals=3).tolist(),
          "means",b.mean(1).round(decimals=3).tolist())
print("Final action-related names",[(n,SHAPES[n]) for n in NAMES if
                                   n.startswith("action_out") or EP+"norm.dense" in n])
print("Probe13 output keys:")
q13=json.loads((W/"probe_13"/"results.json").read_text())
print("records",len(q13),str(q13[0])[:500])
zz=np.load(local_path(q13[0]["activations"]))
print([(k,zz[k].shape) for k in zz.files if k.startswith("calls::expert")][-6:])
```
