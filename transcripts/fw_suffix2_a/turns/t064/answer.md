The finer scan shows substantial tagged/clean differences even in the closest-matching context bands, especially for Camera, Initial States, and Noise. That is consistent with the tag altering instruction-token embeddings, not just adding distracting tokens. I’ll separate matched context changes from unmatched features and compute a clean/proprioception-protected contrast basis.
```python
# Static clustering of observed context vectors; no network evaluation or weight fitting.
XB=torch.from_numpy(np.load(W/"context_bands_56x30x4096.npy")).float()
def representative_vectors(z, cosine=.985):
    norms=z.norm(dim=1)
    z=z[norms>1.0]
    norms=z.norm(dim=1)
    zn=z/norms[:,None]
    sim=zn@zn.T
    order=torch.argsort(norms,descending=True).tolist()
    chosen=[]
    blocked=torch.zeros(len(z),dtype=torch.bool)
    for j in order:
        if blocked[j]: continue
        chosen.append(j)
        blocked |= sim[j]>cosine
    reps=[]
    for j in chosen:
        mask=sim[j]>.996
        reps.append(z[mask].mean(0))
    return torch.stack(reps)
R=[representative_vectors(torch.cat([XD[i],XB[i]],dim=0)) for i in range(56)]
print("Representatives per input:",[len(z) for z in R])
C=torch.cat(R[24:],dim=0)
Cp=C-(C@Qp)@Qp.T
Cpn=Cp/Cp.norm(dim=1,keepdim=True).clamp_min(1e-6)
Uc,Sc,Vhc=torch.linalg.svd(Cpn,full_matrices=False)
energy=(Sc.square().cumsum(0)/Sc.square().sum())
print("Clean representative count:",len(C),"clean PCA cumulative energy:",
      {k:round(float(energy[min(k-1,len(energy)-1)]),5) for k in [32,64,128,256,512,768,1024]})
pairs=[]
diffs=[]
rmatches=[]
for i in range(24):
    raw,clean=R[i],R[24+i]
    cs=torch.nn.functional.normalize(raw,dim=1)@torch.nn.functional.normalize(clean,dim=1).T
    vals,inds=cs.max(1)
    reverse=cs.argmax(0)
    for j in range(len(raw)):
        k=int(inds[j]);val=float(vals[j])
        if val>.65 and int(reverse[k])==j and .65<float(raw[j].norm()/clean[k].norm())<1.5:
            pairs.append((i,j,k,val))
            diffs.append(raw[j]-clean[k])
            rmatches.append(raw[j])
D=torch.stack(diffs)
print("Mutual matches:",len(pairs),"by category:",
      {cat:sum(items[p[0]]["category"]==cat for p in pairs) for cat in sorted(set(t["category"] for t in items[:24]))})
_,Sd,Vhd=torch.linalg.svd(D,full_matrices=False)
print("Contrast singular values first20:",np.round(Sd[:20].numpy(),3).tolist())
print("Contrast energy rank8/16/32/64:",[round(float(Sd[:k].square().sum()/Sd.square().sum()),4) for k in [8,16,32,64]])
for rank in [128,256,512,768]:
    Q=torch.cat([Qp,Vhc[:min(rank,len(Vhc))].T],dim=1)
    dp=D-(D@Q)@Q.T
    print("Clean rank",rank,"contrast outside clean span fraction",
          round(float(dp.square().sum()/D.square().sum()),4),
          "mean contrast norm",round(float(dp.mean(0).norm()),4))
torch.save(dict(representatives=R,Qp=Qp,C=C,Sc=Sc,Vhc=Vhc,pairs=pairs,D=D,
                Sd=Sd,Vhd=Vhd,raw_matches=torch.stack(rmatches)),W/"dense_context_geometry.pt")
```
