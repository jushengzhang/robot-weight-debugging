I will restore the real token embeddings and measure the controlled language effects layer by layer. I will construct any candidate subspace from paired activation differences only, without success labels, and protect the common visual residual so the edit does not simply amplify the entire scene representation.
```python
drop_edit(en)
p27=W/"probe_27"
r27=json.loads((p27/"results.json").read_text())
z27={arm:[np.load(p27/f"probe_{arm}_{i}_acts.npz") for i in range(64)] for arm in ["orig","edit"]}
print("Expert keys",[(k,z27["orig"][0][k].shape) for k in z27["orig"][0].files if "expert.14" in k])
def get_stack(arm,key):
    return torch.from_numpy(np.stack([z[key] for z in z27[arm]])).float()
print("Controlled swap mean relative residual change:")
for net,ls,groups in [("vlm",[9,11,12,13,14,15,16,17],["base","wrist","lang"]),
                     ("expert",[12,13,14,15,16,17],["action"])]:
    for l in ls:
        out=[]
        for g in groups:
            key=f"mean::{net}.{l}.resid::{g}"
            X=get_stack("orig",key);Y=get_stack("edit",key)
            ratios=(Y-X).norm(dim=-1)/X.norm(dim=-1)
            out.append((g,round(float(ratios.mean()),3),round(float(ratios.median()),3)))
        print(net,l,out)
delta=np.stack([np.load(p27/f"probe_edit_{i}_action.npy")-
                np.load(p27/f"probe_orig_{i}_action.npy") for i in range(64)])
print("XYZ action change RMS",np.round(np.sqrt((delta[:,:,:3]**2).mean(axis=(1,2))),3).tolist()[:16])
print("Median XYZ RMS",round(float(np.median(np.sqrt((delta[:,:,:3]**2).mean(axis=(1,2))))),4))
print("Norm tensor names",[(n,SHAPES[n]) for n in NAMES if LP+"layers.14.input_layernorm" in n])
# Normalize pooled proxies and extract language-caused visual directions.
bases={}; reports={}
for layer in [12,13,14,15,16,17]:
    prv=layer-1
    ng=load_original(LP+f"layers.{layer}.input_layernorm.weight")+1
    norm=lambda X: X/torch.sqrt(X.square().mean(-1,keepdim=True)+1e-6)*ng
    Xs=[];Ys=[]
    for cam in ["base","wrist"]:
        Xs.append(norm(get_stack("orig",f"mean::vlm.{prv}.resid::{cam}")))
        Ys.append(norm(get_stack("edit",f"mean::vlm.{prv}.resid::{cam}")))
    X=torch.cat(Xs);Y=torch.cat(Ys)
    D=Y-X
    # Protect camera means and the top four scene PCs of the paired midpoint.
    mid=(X+Y)/2
    cammean=torch.stack([mid[:64].mean(0),mid[64:].mean(0)])
    centered=mid-torch.cat([cammean[0].expand(64,-1),cammean[1].expand(64,-1)])
    _,_,Vs=torch.linalg.svd(centered,full_matrices=False)
    C=torch.cat([cammean.T,Vs[:4].T],dim=1)
    Q=torch.linalg.qr(C,mode="reduced").Q
    Dr=D-(D@Q)@Q.T
    _,s,Vh=torch.linalg.svd(Dr,full_matrices=False)
    U=Vh[:8].T.contiguous()
    bases[layer]=U
    wn=LP+f"layers.{layer}.self_attn.v_proj.weight"
    V=load_original(wn)
    reports[layer]={"contrast_retained":float(Dr.square().sum()/D.square().sum()),
                    "rank8_retained":float(s[:8].square().sum()/s.square().sum()),
                    "weight_frac":float((V@U).square().sum()/V.square().sum()),
                    "original_projection_frac":float((X@U).square().sum()/X.square().sum())}
print("Subspace diagnostics",json.dumps(reports,indent=2))
torch.save({"basis":bases,"reports":reports,"inputs":[x["id"] for x in lang_inputs]},
           W/"subspaces"/"controlled_grounded_language.pt")
print("Current edits",edited())
```
