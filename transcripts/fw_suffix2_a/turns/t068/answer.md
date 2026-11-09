The projection barely changed tagged actions and did not improve the overall gap, so I’ll discard it. Next I’ll test the added-token route more selectively: derive gate detectors from unmatched tagged features while protecting both clean features and matched tagged-prefix features. This addresses a weakness of the earlier gates, which could also suppress useful instruction content.
```python
clear_edits()
with (W/"notes.md").open("a") as f:
    f.write("\nPass19 dense prefix rank16 rejected: gap ratio1.0052, clean drift.00105, paraphrase.00136. "
            "Matched-prefix contrast removal has little causal effect; all edits cleared. Next selective unmatched-token gate protects raw prefix as well as clean text.\n")
raw_prefix=[];novel=[];novel_meta=[]
for i in range(24):
    raw,clean=R[i],R[24+i]
    sim=torch.nn.functional.normalize(raw,dim=1)@torch.nn.functional.normalize(clean,dim=1).T
    vals=sim.max(1).values
    raw_prefix.append(raw[vals>.65])
    novel.append(raw[vals<.55])
    novel_meta.extend([(i,items[i]["category"])]*int((vals<.55).sum()))
RP=torch.cat(raw_prefix)
T=torch.cat(novel)
print("Matched-prefix representatives:",len(RP),"unmatched:",len(T))
RPo=RP-(RP@Q)@Q.T
_,Sp,Vhp=torch.linalg.svd(RPo,full_matrices=False)
print("Additional raw-prefix PCA energy:",{k:round(float(Sp[:k].square().sum()/Sp.square().sum()),4) for k in [64,128,256,384]})
Qprotect=torch.linalg.qr(torch.cat([Q,Vhp[:256].T],dim=1),mode="reduced").Q
To=T-(T@Qprotect)@Qprotect.T
_,St,Vht=torch.linalg.svd(To,full_matrices=False)
print("Unmatched residual energy:",round(float(To.square().sum()/T.square().sum()),4),
      "top8/16 energy:",[round(float(St[:k].square().sum()/St.square().sum()),4) for k in [8,16]])
cats=sorted(set(t["category"] for t in items[:24]))
proto=[]
for cat in cats:
    mask=torch.tensor([x[1]==cat for x in novel_meta])
    tv=To[mask].mean(0)
    print(cat,"unmatched",int(mask.sum()),"mean residual norm",round(float(tv.norm()),4))
    proto.append(tv/tv.norm().clamp_min(1e-8))
dirs=torch.stack(proto)
score_clean=C@dirs.T
score_pre=RP@dirs.T
score_tag=T@dirs.T
print("Prototype max clean score:",np.round(score_clean.max(0).values.numpy(),3).tolist())
print("Prototype max raw-prefix score:",np.round(score_pre.max(0).values.numpy(),3).tolist())
print("Prototype tag score median/max:",qnt(score_tag.max(1).values))
torch.save(dict(dirs=dirs,Qprotect=Qprotect,raw_prefix=RP,novel=T,
                novel_meta=novel_meta,Vht=Vht,St=St,categories=cats),W/"dense_gate_geometry.pt")

gm=torch.load(W/"prototype_gate_metadata.pt",weights_only=False)
print("Old gate specs fields:",{e:list(s.keys()) for e,s in gm["specs"].items()})
for ex in ["action","video"]:
    s=gm["specs"][ex]
    print(ex,{k:(tuple(t.shape) if torch.is_tensor(t) else str(t)[:100]) for k,t in s.items() if k!="coords"})
    p=f"mot.mixtures.{ex}.blocks.0.cross_attn."
    olddir=W/"fwd"/"probe_11_weights"
    qb=torch.from_numpy(np.load(olddir/(p+"q.bias.npy")))
    kw=torch.from_numpy(np.load(olddir/(p+"k.weight.npy")))
    changed=(qb!=load_original(p+"q.bias")).nonzero().flatten()
    nq=load_original(p+"norm_q.weight"); nk=load_original(p+"norm_k.weight")
    print(" First gate query coordinate:",int(changed[0]),"bias",float(qb[changed[0]]),
          "q/k scales",float(nq[changed[0]]),float(nk[changed[0]]),
          "key row norm",float(kw[changed[0]].norm()))
```
