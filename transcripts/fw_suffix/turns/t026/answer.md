I will construct an orthogonal nuisance projector from the final residuals, protecting the clean residual span across all ten denoising steps. The edit will change only the action head’s sensitivity to the leading contrast directions; it will not fit target actions or modify the transformer blocks.
```python
p9=json.loads((work/"probe_9/results.json").read_text())
p9acts=[dict(np.load(r["activations"])) for r in p9]
print("Residual key examples:",[(k,v.shape) for k,v in list(p9acts[0].items())[:10]])
resid_bases={}
for l in [0,1,3,7,12,18,23,29]:
    key=f"calls::action.{l}.resid::action"
    R=np.concatenate([p9acts[2*i][key] for i in range(24)],axis=0)
    C=np.concatenate([p9acts[2*i+1][key] for i in range(24)],axis=0)
    U,Q,D,Ct,st=contrast_basis(R,C,rank=16)
    resid_bases[l]=(U,Q,D,Ct,st)
    print("Layer",l,"D-rms",round(st["contrast_rms"],5),
          "protect",st["protected_rank"],
          "D-remain",round(st["remaining_contrast_frac"],4),
          "D-top16",round(st["top4_contrast_frac"],4))
head_name="mot.mixtures.action.head.weight"
H=load_original(head_name)
U=resid_bases[29][0]
print("Final readout projection diagnostics:")
for rank in [1,2,4,8,16]:
    u=U[:,:rank]
    print(rank,"relative head delta",round(float((H@u).norm()/H.norm()),5),
          "contrast captured",round(float((resid_bases[29][2]@u).square().sum()/
                                         resid_bases[29][2].square().sum()),4))
clear_edits()
rank_head=8
Uh=U[:,:rank_head]
save_edit(head_name,H-(H@Uh)@Uh.T)
torch.save({"U":U,"Q":resid_bases[29][1],"stats":resid_bases[29][4]},work/"head_contrast_basis.pt")
config={"name":"action_head_P8","module":head_name,"rank":8,"alpha":1.,
        "protected":"all 240 clean denoising-step mean residuals"}
(work/"current_edit.json").write_text(json.dumps(config,indent=2))
print("Current edit:",list(edited()))
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 9: original residual reference\n"
            "48 raw/strip inputs, layers 0,1,3,7,12,18,23,29, per-denoising-step means.\n"
            "Next candidate action_head_P8: right-project action head weight away from "
            "8 leading tag contrast directions orthogonal to the span of 240 clean residual means. "
            "Only one tensor changed; static orthogonal erasure, no action-target fitting.\n")
```
