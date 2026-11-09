I will assemble the full context vectors, remove all temporary edits, and examine two distinct effects: novel tag-like vectors and changes to vectors that still closely match clean task context. This distinction matters because masking suffix tokens cannot undo contextual changes already made by UMT5.
```python
b4=extract_band_slice(9,3584)
X=np.concatenate([np.load(W/f"band_slice_{st:04d}.npy") for st in (0,896,1792,2688,3584)],axis=-1)[:,:,:4096]
np.save(W/"context_bands_56x30x4096.npy",X)
clear_edits()
print("INSTRUMENTATION REMOVED; EDITS:",len(edited()))
Xt=torch.from_numpy(X)
xp=torch.einsum("nbd,d->nb",Xt,u_prior)
norms=Xt.norm(dim=-1)
print("u SCORE AT BANDS, example raw:",[round(float(v),3) for v in xp[0]])
print("u SCORE AT BANDS, example clean:",[round(float(v),3) for v in xp[24]])
Xn=Xt/(norms[:,:,None]+1e-9)
similarity=torch.einsum("ibd,icd->ibc",Xn[:24],Xn[24:48])
match_cos,match_id=similarity.max(-1)
for cat in tagcats:
    ids=[i for i,p in enumerate(splits["diagnostic"][:24]) if p["category"]==cat]
    print(cat,"mean nearest-clean cosine by band groups",
          [round(float(match_cos[ids,a:b].mean()),3) for a,b in ((0,6),(6,12),(12,18),(18,24),(24,30))])
# Covariance/SVD analysis of measured clean context, not a model forward.
prop=torch.cat([load_original("proprio_encoder.weight"),load_original("proprio_encoder.bias")[:,None]],1)
Qprop=torch.linalg.qr(prop,mode="reduced").Q
cleanvec=Xt[24:].reshape(-1,4096)
clean_nop=cleanvec-(cleanvec@Qprop)@Qprop.T
Uc,Sc,Vhc=torch.linalg.svd(clean_nop,full_matrices=False)
print("CLEAN BAND SINGULAR VALUES:",[(i,round(float(Sc[i]),4)) for i in (0,1,2,5,10,20,40,80,120,180,240,360,500,700,900)])
print("CLEAN ENERGY FRACTIONS:",[(r,round(float(Sc[:r].square().sum()/Sc.square().sum()),5)) for r in (32,64,128,192,256,384)])
torch.save({"X":Xt,"u_scores":xp,"norms":norms,"match_cos":match_cos,"match_id":match_id,
            "Qprop":Qprop,"Sc":Sc,"Vhc":Vhc},W/"context_band_analysis.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n### Passes7-9 complete context-band acquisition\ncontext_bands_56x30x4096.npy saved (24raw+24clean+8language). Readout time variation0 for all slices. ALL426 instrumentation edits removed. Subsequent analysis uses observed vectors only.\n")
```
