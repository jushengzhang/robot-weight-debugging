The repair reduces the normalized action gap by about 9%, but paraphrase drift is noticeably larger than clean drift. I will inspect which sampled context features it changes and how much clean-language structure lies outside the protected span before increasing its strength.
```python
X = torch.from_numpy(np.load(W/"context_features_72x30x4096.npy")).double()
Qp0 = torch.linalg.qr(torch.cat([load_original("proprio_encoder.weight").double(),
                                load_original("proprio_encoder.bias").double()[:,None]],dim=1)).Q
C = X[24:].reshape(-1,4096)
Cperp = C - (C@Qp0)@Qp0.T
_, Cs, Cv = torch.linalg.svd(Cperp,full_matrices=False)
print("CLEAN CONTEXT SINGULARS", {i:round(float(Cs[i]),6) for i in [0,7,31,127,255,383,511,639,767,1023,1279,1439]})
ce = torch.cumsum(Cs**2,0)/torch.sum(Cs**2)
print("CLEAN ENERGY RANKS", {f:int(torch.searchsorted(ce,torch.tensor(f))+1) for f in [.9,.99,.999,.9999,.99999,.999999]})
U = proj["U"][:,:64].double()
B = proj["B_nearest"][:,:64].double()
change = (X@U)@B.T
print("MEASURED CONTEXT CHANGE FRACTIONS",
      {g:float(torch.linalg.norm(change[s])/torch.linalg.norm(X[s]))
       for g,s in [("raw",slice(0,24)),("clean",slice(24,48)),("language",slice(48,72))]})
for j in [0,1,2,3,14,15,16,29]:
    print("query",j, meta["query_specs"][j], "norm",
          [round(float(torch.linalg.norm(X[s,j],dim=-1).mean()),4) for s in [slice(0,24),slice(24,48),slice(48,72)]])
a18 = actions_for(18,"action_norm")
print("LARGEST LANGUAGE DRIFTS")
ds=np.sqrt(np.mean((a18[48:]-An0[48:])**2,axis=(1,2)))
for j in np.argsort(ds)[::-1][:6]:
    print(int(j), round(float(ds[j]),5), sp["language_diag"][int(j)]["name"])
print("D MATCH RESIDUAL", float(torch.linalg.norm(proj["D"].double()@proj["U"][:,:64].double()
          - proj["D"].double()@proj["U"][:,:64].double())))
torch.save({"Qproprio":Qp0.float(),"Csingular":Cs.float(),"Cright":Cv.float()},W/"clean_context_svd.pt")
with (W/"notes.md").open("a") as f:
    f.write("\nPass18 rank64 oblique both: normalized baseline gap .072859 -> .066593 (ratio .914). Executed gap ratio .919; clean drift .00359, language drift .02813. No rollout validation yet. Need stronger language protection before retention.\n")
```
