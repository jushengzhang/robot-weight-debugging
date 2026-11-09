The leading four contrast modes are substantial, and the projector checks pass numerically. I’ll test a rank-4 correction at each text MLP’s output. I’ll also protect the two directions altered by the readout itself, so the candidate does not depend on those instrumentation changes.
```python
# Recompute with exact protection against the two text-readout substitutions.
prefix_bases={}
for ex in ("action","video"):
    T=pc[ex].double()
    pre=f"mot.mixtures.{ex}.text_embedding."
    w0=load_original(pre+"0.weight").double()
    b0=load_original(pre+"0.bias").double()
    w2=load_original(pre+"2.weight").double()
    b2=load_original(pre+"2.bias").double()
    h,j=gate["designs"][ex]["hidden"],gate["designs"][ex]["channel"]
    ej=torch.zeros(w2.shape[0],1,dtype=torch.float64); ej[j]=1
    protect=torch.cat([T[clean_idx].T,w2@(w0@Pctx.double()),
                       (w2@b0)[:,None],b2[:,None],w2[:,h:h+1],ej],dim=1)
    Q,Sq,_=torch.linalg.svd(protect,full_matrices=False)
    Q=Q[:,Sq>Sq[0]*1e-7]
    d=(T[:48:2]-T[1:48:2]).T
    dp=d-Q@(Q.T@d)
    U,S,Vh=torch.linalg.svd(dp,full_matrices=False)
    B=(d@Vh.T)/S[None,:]
    prefix_bases[ex]={"U":U.float(),"B":B.float(),"S":S.float(),
                       "Q":Q.float(),"D":d.float()}
torch.save(prefix_bases,W/"masked_prefix_projectors.pt")

def apply_prefix_projector(rank=4, strength=1.0, experts=("action","video"), oblique=True):
    for n in list(edited()): drop_edit(n)
    rel={}
    for ex in experts:
        d=prefix_bases[ex]
        U=d["U"][:,:rank]
        B=d["B"][:,:rank] if oblique else U
        n=f"mot.mixtures.{ex}.text_embedding.2.weight"
        x=load_original(n)
        dx=B@(U.T@x)
        save_edit(n,x-strength*dx)
        rel[ex]=float(strength*dx.norm()/x.norm())
    meta={"name":"masked_prefix_output_projection","rank":rank,
          "strength":strength,"experts":list(experts),"oblique":oblique,
          "relative_weight_changes":rel,"edited_names":list(edited())}
    (W/"current_edit.json").write_text(json.dumps(meta,indent=2))
    return meta

meta=apply_prefix_projector(rank=4,strength=1.0)
(W/"candidate_prefix_rank4.json").write_text(json.dumps(meta,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass30 masked-context readout completed\n"
            "Instrumentation fully removed. Readout time variation: action2.1e-5 RMS, video0. "
            "Masked post-text-MLP raw-clean mean contrast norms: action.128,video.311. "
            "Rank4 oblique output projector now applied to text_embedding.2.weight in both "
            "experts only. Protection includes56 clean/paraphrase means, linear proprio span, "
            "bias directions, and exact two-direction readout substitution span. No training; "
            "static SVD projector. Candidate awaits probe31.\n")
print(json.dumps(meta,indent=2))
```
