I will compare all three route tests and inspect the prior gate’s exact tensor changes. A possible next test is to combine context correction with suffix suppression, since a corrected suffix token may still attract attention even after its embedding is changed.
```python
m21=probe_metrics(21)
print(json.dumps(m21,indent=2))
snapshot_candidate("oblique_r256_d1_video")
print("PRIOR GATE TEXT MODIFICATIONS")
for ex in ["action","video"]:
    for suf in ["0.weight","0.bias","2.weight","2.bias"]:
        n=f"mot.mixtures.{ex}.text_embedding.{suf}"
        a=torch.from_numpy(np.load(W/"prior_candidate"/(n+".npy")))
        d=a-load_original(n)
        if d.ndim==2:
            rows=torch.where(torch.any(d!=0,dim=1))[0]
            cols=torch.where(torch.any(d!=0,dim=0))[0]
            print(ex,suf,"rows",len(rows),rows[:8].tolist(),"cols",len(cols),cols[:8].tolist(),
                  "max",float(d.abs().max()))
        else:
            ids=torch.where(d!=0)[0]
            print(ex,suf,"idx",ids.tolist(),"values",a[ids].tolist())
print("EXAMPLE PRIOR GATE TENSORS")
pn=sorted(p.stem for p in (W/"prior_candidate").glob("*.npy"))
print(pn[:10])
for n in pn:
    if "action.blocks.0.cross_attn" in n:
        print(n, SHAPES.get(n))
```
