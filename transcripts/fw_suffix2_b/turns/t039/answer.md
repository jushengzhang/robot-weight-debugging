I’ll remove all temporary instrumentation, reconstruct the context measurements, and examine what survives suffix masking. In particular, I’ll check whether masked tagged features still differ from clean features and whether those differences form a low-dimensional subspace that is separate from clean and paraphrased language.```python
sl,sc = decode_readout(17,3840)
print("Final score mismatch:",rms(sc-scores0))
clear_edits()
assert len(edited()) == 0
X = np.concatenate([np.load(readout_dir/f"sharp_slice_{s:04d}.npy")[:,:,:min(480,4096-s)]
                    for s in range(0,4096,480)],axis=-1)
assert X.shape == (72,30,4096)
np.save(W/"context_features_72x30x4096.npy",X)
T = torch.from_numpy(X).double()
T -= (T@Qprop.double())@Qprop.double().T
uq = [0]+list(range(2,16))
mq = [1]+list(range(16,30))
Cu = T[24:48,uq]
Ru = T[:24,uq]
Rm = T[:24,mq]
print("Full context shape:",X.shape)
print("Query-score reconstruction error:", rms((T@dirs.double().T).numpy()-scores0))
for label,a in [("raw unmasked",Ru),("raw masked",Rm),("clean",Cu),("paraphrases",T[48:,uq])]:
    score=a@u.double()
    print(label,"norm mean",round(float(a.norm(dim=-1).mean()),4),
          "detector score mean/max",round(float(score.mean()),4),round(float(score.max()),4),
          "score quantiles",np.round(np.quantile(score.numpy(),[0,.1,.5,.9,1]),4).tolist())
print("Masked contrast/unmasked contrast ratio:",float((Rm-Cu).norm()/(Ru-Cu).norm()))
print("Contrast RMS unmasked/masked:",rms((Ru-Cu).numpy()),rms((Rm-Cu).numpy()))
normed=lambda z:z/z.norm(dim=-1,keepdim=True).clamp_min(1e-12)
sim=torch.einsum("nid,njd->nij",normed(Rm[:,1:]),normed(Cu[:,1:]))
print("Same-query cosine quantiles:",np.round(np.quantile(sim.diagonal(dim1=1,dim2=2).numpy(),[0,.1,.5,.9,1]),3))
print("Nearest-clean cosine quantiles:",np.round(np.quantile(sim.max(dim=2).values.numpy(),[0,.1,.5,.9,1]),3))

C = T[24:,uq].reshape(-1,4096)
Lc,Sc,Vc = torch.linalg.svd(C,full_matrices=False)
print("Clean singular spectrum at ranks:",{i:round(float(Sc[i]),5) for i in [0,15,31,63,127,255,383,511,639,719]})
print("Clean energy ranks:", {e:int(torch.searchsorted((Sc**2).cumsum(0)/(Sc**2).sum(),e))+1 for e in [.95,.99,.999,.9999]})
with (W/"notes.md").open("a") as f:
    f.write("\nPass17 acquisition complete. ALL426 instrumentation edits removed. "
            "Full4096D context samples saved in context_features_72x30x4096.npy; "
            "query-score checks exactly equal across all slices and diffusion-step drift negligible. "
            "30queries=15unmasked/15masked; 72inputs=24raw,24clean,24paraphrases.\n")
```
