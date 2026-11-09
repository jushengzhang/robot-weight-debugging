I will rank the later heads by three observable properties: attention to texture-changed regions, movement of attention between paired textures, and concentration of that attention. I will only consider a head-local query-normalization edit if these maps identify a specific texture-sensitive reader.
```python
p17=read_probe(17)
late_map_stats={}
for L in [10,14,20,29]:
    data=[]
    for j in [0,2]:
        a,b=p17[j],p17[j+1]
        M=np.load(a["attention"][str(L)]).mean(1)
        N=np.load(b["attention"][str(L)]).mean(1)
        mass=(M.sum(-1)+N.sum(-1))/2
        P=M/M.sum(-1,keepdims=True).clip(1e-12)
        Q=N/N.sum(-1,keepdims=True).clip(1e-12)
        d=token_difference(a["input"],b["input"])
        mask=d>np.quantile(d,.6)
        enrich=((P+Q)/2)[:,mask].sum(-1)/mask.mean()
        tv=np.abs(P-Q).sum(-1)/2
        ent=-(0.5*(P+Q)*np.log(0.5*(P+Q)+1e-12)).sum(-1)/np.log(98)
        peak=np.maximum(P.max(-1),Q.max(-1))
        data.append(np.stack([mass,enrich,tv,ent,peak],1))
    late_map_stats[L]=np.stack(data)
    H=np.mean(data,0)
    print("\nL",L,"head,mass,enrichment,TV,entropy,peak")
    for h in np.argsort(-(H[:,1]*H[:,2]))[:8]:
        print(h,np.round(H[h],3).tolist(),
              "pair enrich",np.round(late_map_stats[L][:,h,1],2).tolist())
np.savez(W/"late_attention_map_stats.npz",
         **{f"L{L}":v for L,v in late_map_stats.items()})
print("No edits:",edited())
```