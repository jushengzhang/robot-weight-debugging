I will remove all diagnostic edits, assemble the three feature slices, and examine which directions distinguish tagged contexts from clean and paraphrased contexts. In particular, I will check whether the residual after masking is genuinely low-rank and whether it overlaps useful clean features.
```python
f10=extract_features(10,2)
clear_edits()
Z=torch.zeros(128,3,3072,dtype=torch.float64)
for sl in range(3):
    Z[:,:,slices_v[sl]]=torch.from_numpy(feature_reads[sl]).double()
torch.save({"features":Z,"feature_ids":feature_ids,"excluded":excluded_v},W/"video_context_features.pt")
cleanF=Z[48:,0]
diffU=Z[:48,0]-Z[48:96,0]
diffM=Z[:48,1]-Z[48:96,0]
emph=Z[:48,2]
print("ALL TEMPORARY EDITS CLEARED:",len(edited()))
print("FULL FEATURE GAPS uniform/masked:",float(diffU.norm(dim=1).mean()),float(diffM.norm(dim=1).mean()))
for lab,D in (("uniform",diffU),("masked",diffM)):
    U,S,Vh=torch.linalg.svd(D,full_matrices=False)
    print(lab,"SVD energy cum1/2/4/8/16/32:",[round(float((S[:k]**2).sum()/(S*S).sum()),4) for k in (1,2,4,8,16,32)])
    cenergy=(cleanF*cleanF).mean(0).sqrt()
    denergy=(D*D).mean(0).sqrt()
    score=denergy/(cenergy+1e-4)
    inds=score.argsort(descending=True)[:20]
    print("TOP SELECTIVITY FEATURES (id,ratio,contrast,clean):",
          [(int(i),round(float(score[i]),2),round(float(denergy[i]),4),round(float(cenergy[i]),4)) for i in inds])
    inds=denergy.argsort(descending=True)[:15]
    print("TOP CONTRAST FEATURES:",[(int(i),round(float(score[i]),2),round(float(denergy[i]),4)) for i in inds])
# Clean/proprio-protected feature geometry, not a fitted loss.
# Include the linear image of the entire proprio subspace through original input projection,
# the bias direction, and all measured clean/paraphrase feature means.
protect=torch.cat([cleanF.T, (v_w1@Qp).double(),v_b1[:,None].double()],dim=1)
Uc,Sc,_=torch.linalg.svd(protect,full_matrices=False)
C=Uc[:,Sc>Sc[0]*1e-8]
print("PROTECTED RANK:",C.shape[1])
for lab,D in (("uniform",diffU),("masked",diffM)):
    R=D-(D@C)@C.T
    _,s,vh=torch.linalg.svd(R,full_matrices=False)
    print(lab,"orthogonal retained energy",round(float((R*R).sum()/(D*D).sum()),4),
          "detector singular vals",[round(float(x),4) for x in s[:8]])
with (W/"notes.md").open("a") as f:
    f.write("\nPass10 complete. All48 diagnostic tensors cleared. Readout data saved video_context_features.pt "
            "(128 inputs x uniform/masked/emphasized x3072 hidden channels;3 lowest-energy channels omitted). "
            "Candidate construction must use static linear algebra only, not loss fitting.\n")
```
