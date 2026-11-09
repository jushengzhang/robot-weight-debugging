Suffix masking left essentially all of the context discrepancy: the masked/unmasked contrast-norm ratio was 1.003. The masked tagged features also retained a positive tag-detector shift absent from clean and paraphrased features. I’ll therefore test a projection at the shared text-input stage, protecting the observed clean-language subspace and removing leading tag-associated directions. This targets contextual contamination as well as explicit suffix tokens.```python
# Protect the clean/paraphrase feature span and the exact proprio span.
protect_rank = 512
Qclean0 = Vc[:protect_rank].T
Qclean0 -= Qprop.double()@(Qprop.double().T@Qclean0)
Qprotect = torch.linalg.qr(torch.cat([Qprop.double(), Qclean0],dim=1),mode="reduced").Q
# Two banks of corrupted observations: all context and suffix-masked context.
Rbank = torch.cat([Ru,Rm],dim=1)
targets = []
cosines = []
for i in range(24):
    s = normed(Rbank[i,1:]) @ normed(Cu[i,1:]).T
    best_sim,best_idx = s.max(1)
    targ = Cu[i,1:][best_idx].clone()
    targ[best_sim < .5] = 0  # novel/suffix-like contexts: no invented semantic target
    # The uniform queries have known raw-clean correspondence.
    target = torch.cat([Cu[i,:1],targ],dim=0)
    target[15] = Cu[i,0]
    targets.append(target)
    cosines.append(best_sim)
Target = torch.stack(targets)
D0 = (Rbank-Target).reshape(-1,4096)
scale = Rbank.norm(dim=-1).reshape(-1).clamp_min(.1)
D = D0 / scale[:,None]
Dp = D - (D@Qprotect)@Qprotect.T
Ld,Sd,Vd = torch.linalg.svd(Dp,full_matrices=False)
Uall = Vd.T
Ball = (D.T@Ld)/Sd.clamp_min(1e-10)[None,:]
print("Protected span:",Qprotect.shape)
print("Contrast orthogonal fraction:",float(Dp.norm()/D.norm()))
print("Tag singular values:", {i:round(float(Sd[i]),5) for i in [0,7,15,31,63,127,255,383,511,639]})
print("Contrast energy ranks:",{e:int(torch.searchsorted((Sd**2).cumsum(0)/(Sd**2).sum(),e))+1 for e in [.5,.8,.9,.95,.99]})
for rank in [16,32,64,128,256]:
    print("rank",rank,"oblique basis RMS norm",round(float(Ball[:,:rank].norm()/math.sqrt(rank)),3),
          "identity error",float((Uall[:,:rank].T@Ball[:,:rank]-torch.eye(rank)).abs().max()),
          "protected leakage",float((Uall[:,:rank].T@Qprotect).abs().max()))
torch.save({"U":Uall.float(),"B_nearest":Ball.float(),"S":Sd.float(),
            "Qprotect":Qprotect.float(),"D":D.float(),"protect_rank":protect_rank},
           W/"rich_context_projection.pt")

def apply_context_projector(rank=64,dose=1.,kind="oblique",experts=("action","video")):
    clear_edits()
    U = Uall[:,:rank].float()
    B = (Ball if kind=="oblique" else Uall)[:,:rank].float()
    for ex in experts:
        n=f"mot.mixtures.{ex}.text_embedding.0.weight"
        w=load_original(n)
        delta=(w@B)@U.T
        save_edit(n,w-dose*delta)
        print(ex,kind,"rank",rank,"dose",dose,"relative delta",float((dose*delta).norm()/w.norm()),
              "proprio error",float((delta@prop).norm()/(w@prop).norm()))
    meta={"rank":rank,"dose":dose,"kind":kind,"experts":list(experts)}
    (W/"current_candidate.json").write_text(json.dumps(meta,indent=2))
    return meta
candidate = apply_context_projector(rank=64,dose=1.,kind="oblique")
with (W/"notes.md").open("a") as f:
    f.write("\nNew diagnosis: suffix-masked vs unmasked contrast norm ratio1.0033; "
            "masked tagged detector score mean.2488 vs clean.0080/paraphrase.0054. "
            "Thus high-confidence suffix suppression leaves substantial broader context corruption.\n"
            "Candidate rich-context oblique rank64: protects512 clean/paraphrase PCs plus9 proprio dimensions. "
            "Contrast modes from30 context queries per tagged input, nearest within-prompt clean sample "
            "(cos<.5 novel samples erased rather than assigned semantic targets); exact static SVD projector, "
            "no training or action-target fitting. Only2 text_embedding.0.weight tensors edited.\n")
```
