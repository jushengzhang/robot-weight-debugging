I’ll identify visual directions changed consistently by the environment perturbation but not by the clean scene variation. The proposed repair is a small projection in the multimodal projector, suppressing only those nuisance directions while leaving the rest of the visual representation and all language weights unchanged.

```python
pr6=json.loads((W/"probe_6"/"results.json").read_text())
p3eids=[groups[g][p]["id"] for g in diag_groups for p in ("task","semantic")]
p3eids += [1496,2628,4249,1138,1571,2822,4811,4441]
p6eids=[groups[g][p]["id"] for p in ("environment","position","object") for g in diag_groups]
initacts={eid:a for eid,a in zip(p3eids,p3acts)}
for eid,r in zip(p6eids,pr6):
    initacts[eid]={k:v.copy() for k,v in np.load(workpath(r["activations"])).items()
                   if k.startswith("mean::") or k.startswith("calls::expert")}
def get_init(eid,branch,l,group):
    return torch.from_numpy(initacts[eid][f"mean::{branch}.{l}.resid::{group}"]).float()

def pooled_postln_proxy(x):
    x=x-x.mean(dim=-1,keepdim=True)
    return x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()*postw+postb

clean_features=[]; pert_features={p:[] for p in ("environment","position","object")}
for g in diag_groups:
    for camera in ("base","wrist"):
        a=get_init(groups[g]["task"]["id"],"vision",26,camera)
        b=get_init(groups[g]["semantic"]["id"],"vision",26,camera)
        clean_features.append((pooled_postln_proxy(a)+pooled_postln_proxy(b))/2)
        for p in pert_features:
            x=get_init(groups[g][p]["id"],"vision",26,camera)
            pert_features[p].append(pooled_postln_proxy(x))
clean_features=torch.stack(clean_features)
pert_features={p:torch.stack(xs) for p,xs in pert_features.items()}
clean_center=clean_features.mean(0)
_,cs,cv=torch.linalg.svd(clean_features-clean_center,full_matrices=False)
protect=torch.linalg.qr(torch.cat([clean_center[:,None],cv[:10].T],1),mode="reduced").Q
envd=pert_features["environment"]-clean_features
envd_clean=envd-(envd@protect)@protect.T
_,es,ev=torch.linalg.svd(envd_clean,full_matrices=False)
env_basis=ev[:4].T.contiguous()
print("environment singular energy:",[round(float(v),3) for v in (es.square()/es.square().sum())[:8]])
for rank in (1,2,4):
    u=env_basis[:,:rank]
    print("rank",rank,"captured contrast energy",
          {p:round(float(((x-clean_features)@u).square().sum()/(x-clean_features).square().sum()),3)
           for p,x in pert_features.items()},
          "clean feature energy",round(float((clean_features@u).square().sum()/clean_features.square().sum()),5))
torch.save({"basis":env_basis,"center":clean_center,"singular_values":es},
           W/"subspaces"/"environment_projector.pt")

PROJ="paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
def environment_edit(beta=0.5,rank=2,reset=True):
    if reset: clear_edits()
    u=env_basis[:,:rank]
    w=load_original(PROJ+".weight")
    b=load_original(PROJ+".bias")
    delta=beta*(w@u)@u.T
    save_edit(PROJ+".weight",w-delta)
    # Affine centering preserves the canonical mean exactly in this approximation.
    save_edit(PROJ+".bias",b+delta@clean_center)
    return {"beta":beta,"rank":rank,
            "relative_weight_change":round(float(delta.norm()/w.norm()),5),
            "bias_change_norm":round(float((delta@clean_center).norm()),5)}
print("candidate:",environment_edit())
snapshot("candidate_environment_r2_b05")
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 6: matched visual diagnosis\n"
            "Collected initial-frame vision residuals for all75 diagnostic cases. "
            "Environment changes much larger than object/position in late vision pooled features. "
            "Positional embedding outliers did NOT coincide with fixed wrist attention peaks; no position edit justified.\n"
            "Candidate environment projector: rank2 nuisance basis from environment-clean contrasts "
            "using post-LN pooled proxies, orthogonal to canonical mean and10 canonical visual PCs. "
            "W'=W-0.5 WUU^T; bias compensates canonical mean. Pending causal rollout.\n")
```
