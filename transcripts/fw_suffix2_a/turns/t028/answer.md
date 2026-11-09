The reconstructed vectors show that low-score context changes too, not just the obvious high-score suffix bands. I will look for closely matched raw/clean vectors and characterize their difference subspace, while protecting the dominant clean-context span and the exact proprioception span. This tests the contextualization route that a token gate leaves untouched.
```python
# Candidate matched-prefix contrasts: observed band vectors with a close clean counterpart.
ii,jj=torch.where((match_cos>.65)&(xp[:24]<.45)&(norms[:24]>1.0))
kk=match_id[ii,jj]
raw_match=Xt[ii,jj]
clean_match=Xt[ii+24,kk]
Dprefix=raw_match-clean_match
print("MATCHED PREFIX BAND PAIRS:",len(ii),
      "cos quantiles:",torch.quantile(match_cos[ii,jj],torch.tensor([0.,.25,.5,.75,1.])).tolist())
print("MATCHES BY CATEGORY:",dict(collections.Counter(splits["diagnostic"][int(i)]["category"] for i in ii)))
print("RAW/CLEAN NORMS:",raw_match.norm(dim=1).mean().item(),clean_match.norm(dim=1).mean().item())
Qclean=torch.cat([Qprop,Vhc[:256].T],1)
# Reorthogonalization makes preservation explicit despite numerical SVD residuals.
Qclean=torch.linalg.qr(Qclean,mode="reduced").Q
Dp=Dprefix-(Dprefix@Qclean)@Qclean.T
Up,Sp,Vhp=torch.linalg.svd(Dp,full_matrices=False)
print("PREFIX CONTRAST singulars:",[round(float(x),4) for x in Sp[:20]])
print("PREFIX CONTRAST energy:",[(r,round(float(Sp[:r].square().sum()/Sp.square().sum()),3)) for r in (1,2,4,8,16,32)])
prefix_B=(Dprefix.T@Up[:,:32])/Sp[:32][None,:]
prefix_U=Vhp[:32].T
print("OBLIQUE left-right deviation:",float((prefix_U.T@prefix_B-torch.eye(32)).abs().max()))
# Novel directions in the clearly separated high-score context, as a separate gate hypothesis.
tagmask=(xp[:24]>.45)&(norms[:24]>1)
tagvec=Xt[:24][tagmask]
tagres=tagvec-(tagvec@Qclean)@Qclean.T
Ut,St,Vht=torch.linalg.svd(tagres,full_matrices=False)
print("TAG NOVELTY SINGULARS:",[round(float(v),3) for v in St[:16]])
torch.save({"B":prefix_B,"U":prefix_U,"Qclean":Qclean,"pairs":torch.stack([ii,jj,kk],1),
            "tagU":Vht[:32].T,"tagS":St,"Dprefix":Dprefix},W/"prefix_and_tag_geometry.pt")
def apply_prefix_projector(rank=8,dose=1.,experts=("action","video")):
    clear_edits()
    B=prefix_B[:,:rank];U=prefix_U[:,:rank]
    for expert in experts:
        name=f"mot.mixtures.{expert}.text_embedding.0.weight"
        w=load_original(name)
        new=w-dose*(w@B)@U.T
        put(name,new)
        print(expert,"relative edit norm",round(float((new-w).norm()/w.norm()),5))
apply_prefix_projector(8,1.)
print("CANDIDATE: prefix-band oblique projector rank8; tensors",len(edited()))
with (W/"notes.md").open("a") as f:
    f.write(f"\n### Candidate prefix-band rank8\n{len(ii)} matched raw/clean band pairs with cosine>.65,u<.45,norm>1. Clean span protects256 leading clean/paraphrase band PCs plus exact9D proprio span. SVD of clean-orthogonal contrasts defines P=I-BU^T, U^TB=I. Both text_embedding.0.weight edited; no optimization or network execution in kernel. Awaiting probe10.\n")
```
