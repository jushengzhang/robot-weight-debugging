The readout is stable: action-context variation is about \(2\times10^{-5}\) RMS across denoising steps, and video-context variation is negligible. Tagged and clean contexts still differ after the high-response tokens are masked. I’ll examine low-rank corrections at the text MLP’s output, while protecting the measured clean/paraphrase means and a linear proprioception span.
```python
# Build static contrast projectors from forward-tool measurements.
prop_names=[n for n in NAMES if "proprio" in n]
print("Proprio tensors:",[(n,SHAPES[n]) for n in prop_names])
Pcols=[]
for n in prop_names:
    x=load_original(n)
    if x.ndim==2 and x.shape[0]==4096:
        Pcols.append(x)
    elif x.ndim==1 and x.shape[0]==4096:
        Pcols.append(x[:,None])
Pctx=torch.cat(Pcols,dim=1)
prefix_bases={}
for ex in ("action","video"):
    T=pc[ex].double()
    pre=f"mot.mixtures.{ex}.text_embedding."
    w0=load_original(pre+"0.weight").double()
    b0=load_original(pre+"0.bias").double()
    w2=load_original(pre+"2.weight").double()
    b2=load_original(pre+"2.bias").double()
    # Weight-composition span protection; this is not a model forward pass.
    linear_prop = w2 @ (w0 @ Pctx.double())
    protect=torch.cat([T[clean_idx].T,linear_prop,(w2@b0)[:,None],b2[:,None]],dim=1)
    q,s,_=torch.linalg.svd(protect,full_matrices=False)
    q=q[:,s>s[0]*1e-7]
    d=(T[:48:2]-T[1:48:2]).T
    dp=d-q@(q.T@d)
    U,S,Vh=torch.linalg.svd(dp,full_matrices=False)
    B=(d@Vh.T)/S[None,:]
    prefix_bases[ex]={"U":U.float(),"B":B.float(),"S":S.float(),
                       "Q":q.float(),"D":d.float()}
    print("\n",ex,"clean protection rank",q.shape[1],
          "contrast retained",round(float(dp.norm()/d.norm()),4))
    print("Singular values:",np.round(S.numpy(),5))
    print("Oblique amplification:",np.round(B.norm(dim=0).numpy(),3))
    for rank in (4,8,12,24):
        Ur=U[:,:rank]; Br=B[:,:rank]
        dw=Br@(Ur.T@w2)
        print("rank",rank,"relative W2 change",round(float(dw.norm()/w2.norm()),4),
              "idempotence error",round(float((Ur.T@Br-torch.eye(rank)).norm()),7),
              "clean mean error",round(float((Br@(Ur.T@T[clean_idx].T)).norm()),7))
torch.save(prefix_bases,W/"masked_prefix_projectors.pt")
print("Edits remain:",len(edited()))
```
