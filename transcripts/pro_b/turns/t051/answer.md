The residuals reveal a stronger structural lead: a few channels dominate their magnitude, especially channel 674 in both cameras and channel 738 in language. I will check whether these channels are harmless normalization offsets or major contributors to the attention keys. If a boundary-token channel drives BOS attention, attenuating only its key contribution would be more selective than weakening whole heads.
```python
dims=[674,738,50,604,2019,1577,1967]
print("Dominant-channel input norm gains and key-column norms:")
for l in [1,3,6,9,12,15,17]:
    g=1+load_original(f"{lm}.layers.{l}.input_layernorm.weight")
    wk=load_original(f"{lm}.layers.{l}.self_attn.k_proj.weight")
    print(l,[(d,round(float(g[d]),3),round(float(wk[:,d].norm()),3)) for d in dims])
print("\nLayer-0 MLP channel injections:")
for l in [0,1,6,12,15,17]:
    out=[]
    for m in ["base","wrist","lang"]:
        X=np.stack([a[f"mean::vlm.{l}.mlp_out::{m}"] for a in
                    [dict(np.load(x["activations"])) for x in pr[13][:30]]])
        out.append((m,[(d,round(float(X[:,d].mean()),2)) for d in dims[:4]]))
    print(l,out)
embname=f"{lm}.embed_tokens.weight"
emb=load_original(embname)
tids=[sp.bos_id(),sp.eos_id()]+sp.encode("pick up the tomato sauce and place it in the basket\n")
print("\nToken embedding dominant channels:")
for i in tids:
    e=emb[i]
    ix=torch.topk(e.abs(),3).indices.tolist()
    tok=sp.id_to_piece(i).encode("ascii","replace").decode("ascii").replace("\n","\\n")
    print(i,tok,"norm",round(float(e.norm()),2),"dims",
          [(d,round(float(e[d]),2)) for d in dims[:4]],
          "top",[(d,round(float(e[d]),2)) for d in ix])
del emb
print("\nLayer 6 pooled key contribution norms:")
l=6
wk=load_original(f"{lm}.layers.{l}.self_attn.k_proj.weight")
g=1+load_original(f"{lm}.layers.{l}.input_layernorm.weight")
for m in ["base","wrist","lang"]:
    X=torch.tensor(np.stack([a[f"mean::vlm.{l-1}.resid::{m}"] for a in act3[:30]]))
    X=X/(X.square().mean(1,keepdim=True)+1e-6).sqrt()*g
    k=X@wk.T
    print(m, "whole",round(float(k.norm(dim=1).mean()),2),
          "single-dim",[(d,round(float((X[:,d,None]*wk[:,d]).norm(dim=1).mean()),2)) for d in dims[:4]])
print("\nPooled lang residual range for channel738 by prompt length:")
for j in range(8):
    r=rec_for_path(pr[3][j]["input"])
    n=1+len(sp.encode(r["instruction"]+"\n"))
    print(n,round(float(act3[j]["mean::vlm.5.resid::lang"][738]),1),
          r["instruction"][:80])
```
