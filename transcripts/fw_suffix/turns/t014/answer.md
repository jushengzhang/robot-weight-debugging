The weight API rejected uniform whole-tensor scaling, so I will not use that intervention. Instead, I will analyze low-rank directions that distinguish tagged from clean attention outputs while protecting the clean-output span. This supports a targeted projection edit rather than broadly weakening an expert.
```python
print("Edits after rejected scaling:", list(edited()))
print("Probe activation key counts:", len(probe4_acts[0]),
      collections.Counter(k.split("::")[1].split(".")[0] for k in probe4_acts[0]))

# Construct contrast directions from forward-tool outputs only.
# No model modules are evaluated here.
def contrast_basis(raw, clean, rank=4, protect_rank=None):
    R = torch.as_tensor(np.asarray(raw), dtype=torch.float32)
    C = torch.as_tensor(np.asarray(clean), dtype=torch.float32)
    _, sc, vc = torch.linalg.svd(C, full_matrices=False)
    if protect_rank is None:
        protect_rank = int((sc > sc.max()*1e-5).sum())
    Q = vc[:protect_rank].T.contiguous()
    D = R-C
    Dp = D - (D@Q)@Q.T
    _, sd, vd = torch.linalg.svd(Dp, full_matrices=False)
    U = vd[:rank].T.contiguous()
    stats = {
        "contrast_rms":float(D.square().mean().sqrt()),
        "protected_rank":protect_rank,
        "remaining_contrast_frac":float(Dp.square().sum()/D.square().sum()),
        "top4_contrast_frac":float((D@U).square().sum()/D.square().sum()),
        "clean_projected_frac":float((C@U).square().sum()/C.square().sum()),
        "singular_top":sd[:6].tolist()
    }
    return U, Q, D, C, stats

basis_data = {}
for expert in ["action","video"]:
    for layer in range(30):
        if expert=="action":
            key=f"mean::action.{layer}.attn_out::action"
            r=np.stack([probe4_acts[2*i][key] for i in range(24)])
            c=np.stack([probe4_acts[2*i+1][key] for i in range(24)])
        else:
            rs,cs=[],[]
            for k in common:
                with np.load(rmap[k]["activations"]) as ar, np.load(cmap[k]["activations"]) as ac:
                    for cam in ["agent","wrist"]:
                        key=f"mean::video.{layer}.attn_out::{cam}"
                        rs.append(ar[key])
                        cs.append(ac[key])
            r,c=np.stack(rs),np.stack(cs)
        basis_data[(expert,layer)] = contrast_basis(r,c,rank=4)

print("Contrast-subspace diagnostics:")
for expert in ["action","video"]:
    for l in [0,3,7,10,12,15,20,23,26,29]:
        U,Q,D,C,st = basis_data[(expert,l)]
        align={}
        for module in ["self_attn","cross_attn"]:
            W=load_original(f"mot.mixtures.{expert}.blocks.{l}.{module}.o.weight")
            align[module]=round(float((U.T@W).square().sum()/W.square().sum()),5)
        print(expert,l,
              "D-remain",round(st["remaining_contrast_frac"],3),
              "D-top4",round(st["top4_contrast_frac"],3),
              "C-top4",f'{st["clean_projected_frac"]:.1e}',
              "weight_energy",align)

torch.save({f"{e}.{l}":{"U":z[0],"Q":z[1],"stats":z[4]}
            for (e,l),z in basis_data.items()}, work/"contrast_bases.pt")
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 4 details / API restriction\n"
            "Action-only controlled attention outputs were returned by probe; "
            "video outputs are available from rollout episode means. "
            "Uniform whole-tensor output scaling was rejected by save_edit, no edits made. "
            "Moving to targeted low-rank contrast-subspace interventions, protecting the "
            "observed clean-output span. No local model execution or training.\n")
```
