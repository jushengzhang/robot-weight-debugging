I will analyze the value-only result and build a combined context repair. Besides the masked-prefix contrast, this repair will include strongly tag-emphasized features and map them toward the padding-feature baseline rather than merely suppressing their attention. I will exclude emphasized readouts that did not actually separate from the unweighted context and protect directions that overlap strongly with clean features.
```python
m15=screen_metrics(15);d15=detailed_screen(15)
# GELU of a learned bias is a weight-derived padding reference, not a local model run on inputs.
b=v_b1.double()
padF=0.5*b*(1+torch.tanh(math.sqrt(2/math.pi)*(b+0.044715*b.pow(3))))
padF[excluded_v]=0
ratio_emph=emph.norm(dim=1)/(Z[:48,0].norm(dim=1)+1e-12)
good_emph=ratio_emph>3
print("TAG-EMPHASIZED READOUTS RETAINED:",int(good_emph.sum()),"/48")
for cat in sorted(set(screen_cats)):
    ix=torch.tensor([i for i,c in enumerate(screen_cats) if c==cat])
    print(cat,"emphasis median",round(float(ratio_emph[ix].median()),2),
          "retained",int(good_emph[ix].sum()),"/",len(ix))
protect_union=torch.cat([protect,padF[:,None]],dim=1)
uc,sc,_=torch.linalg.svd(protect_union,full_matrices=False)
Cunion=uc[:,sc>sc[0]*1e-8]
Dunion=torch.cat([diffU,diffM,(emph[good_emph]-padF)/8.0],dim=0)
_,sunion,vunion=torch.linalg.svd(Dunion,full_matrices=False)
print("COMBINED CONTRAST SINGULARS:",[round(float(sunion[i]),6) for i in (0,10,30,60,90,len(sunion)-1) if i<len(sunion)])
# Keep well-measured directions, then drop near-intersections with the protected subspace.
initial_rank=int((sunion>sunion[0]*1e-3).sum())
BB=vunion[:initial_rank].T
G=BB.T@(BB-Cunion@(Cunion.T@BB))
gg,rr=torch.linalg.eigh(G)
keep=gg>0.1
Bunion=BB@rr[:,keep]
Ddet_union=(Bunion-Cunion@(Cunion.T@Bunion))/gg[keep][None,:]
print("COMBINED RANK initial/retained:",initial_rank,Bunion.shape[1],
      "minimum clean-orthogonal fraction",float(gg[keep].min()))
def apply_union(dose=1.0):
    clear_edits()
    n=vd+"text_embedding.2.weight"
    orig=load_original(n).double()
    out=orig-dose*(orig@Bunion)@Ddet_union.T
    save_edit(n,out.float())
    print("UNION PROJECTOR dose",dose,"rank",Bunion.shape[1],
          "relative change",float((out-orig).norm()/orig.norm()),
          "protected mean error",float((cleanF@(out-orig).T).norm()))
torch.save({"B":Bunion,"D":Ddet_union,"C":Cunion,"pad":padF,"good_emph":good_emph,
            "initial_rank":initial_rank},W/"union_projector.pt")
apply_union(1.0)
with (W/"notes.md").open("a") as f:
    f.write("\nCandidate video_union: union of all observed unweighted and masked feature contrasts plus high-confidence "
            "tag-emphasized feature directions relative to padding. SVD and principal-angle filtering only; no optimization. "
            f"Rank{Bunion.shape[1]}, protected clean/proprio/bias/padding span. One video text output tensor.\n")
```
