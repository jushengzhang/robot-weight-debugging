One text-context coordinate, 3639, has a much larger tagged–clean contrast than most coordinates. I’ll check how strongly the text projections use it before treating it as a candidate cause; a large activation alone is not enough evidence.
```python
u=gate["u"]
diag_coords = torch.argsort(D.square().sum(0), descending=True)[:20]
print("Largest raw-clean coordinate energies:")
for i in diag_coords:
    print(int(i), "delta_rms", round(float(rmsd[i]),5),
          "clean_rms", round(float(rmsc[i]),5),
          "mean_detector", round(float(u[i]),5))
for ex in ("action","video"):
    A=load_original(f"mot.mixtures.{ex}.text_embedding.0.weight")
    cn=A.norm(dim=0)
    rank = (-cn).argsort().argsort()
    print("\n",ex,"input-column norm quantiles",
          np.round(torch.quantile(cn,torch.tensor([0.,.5,.9,.99,1.])).numpy(),4))
    print([(int(i),round(float(cn[i]),4),int(rank[i])+1) for i in diag_coords[:12]])
    for i in (3639,):
        aa=A[:,i].abs()
        print("Top rows for column",i,[(int(j),round(float(A[j,i]),4))
                                       for j in aa.topk(10).indices])
print("\nCoordinate 3639 by category:")
for c in sorted(set(cats)):
    ids=[i for i,x in enumerate(cats) if x==c]
    print(c,"raw",np.round(C[raw_idx,3639][ids].numpy(),4),
          "clean",np.round(C[list(range(1,48,2)),3639][ids].numpy(),4))
# Confirm the exact saved gate formula, without running the model.
def saved_weight(passno,n):
    return torch.from_numpy(np.load(W/f"fwd/probe_{passno}_weights"/(n+".npy"))).float()
for ex in ("action","video"):
    des=gate["designs"][ex]
    h,j=des["hidden"],des["channel"]
    pre=f"mot.mixtures.{ex}."
    w0=saved_weight(25,pre+"text_embedding.0.weight")
    b0=saved_weight(25,pre+"text_embedding.0.bias")
    w2=saved_weight(25,pre+"text_embedding.2.weight")
    b2=saved_weight(25,pre+"text_embedding.2.bias")
    print("\nSaved gate",ex,"h,j",h,j,"row gain",float(w0[h]@u),
          "row_error",float((w0[h]-(w0[h]@u)*u).norm()),
          "b0",float(b0[h]),"W2[j,h]",float(w2[j,h]),"b2[j]",float(b2[j]),
          "hidden_col_nonzero",int(torch.count_nonzero(w2[:,h])),
          "channel_row_nonzero",int(torch.count_nonzero(w2[j])))
    p=pre+"blocks.0.cross_attn."
    co=des["layers"][0]["coords"][:3]
    for s in ("q.bias","norm_q.weight","norm_k.weight","k.bias"):
        x=saved_weight(25,p+s)
        print(s,[(i,float(x[i])) for i in co])
    kk=saved_weight(25,p+"k.weight")
    print("k gate entries",[(i,float(kk[i,j]),int(torch.count_nonzero(kk[i]))) for i in co])
```
