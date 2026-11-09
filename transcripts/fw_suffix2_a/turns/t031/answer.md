The contextualization projector increased tagged-action error by about 9.8%, so I will discard it. The band data provide a cleaner alternative: directions that are strong in tag-like context but nearly absent from measured clean context. I will assess a small detector bank along those directions before testing a selective attention gate.
```python
clear_edits()
tag_dirs=Vht[:16].T.contiguous()
scores=torch.einsum("nbd,dk->nbk",Xt,tag_dirs)
maxabs=scores.abs().amax(-1)
for cat in tagcats:
    ids=[i for i,p in enumerate(splits["diagnostic"][:24]) if p["category"]==cat]
    print(cat,"max-abs score bands low/mid/high",
          [[round(float(z),3) for z in torch.quantile(maxabs[ids,a:b].flatten(),torch.tensor([.25,.5,.9]))]
           for a,b in ((0,12),(12,18),(18,30))])
print("clean/lang max abs quantiles:",torch.quantile(maxabs[24:].flatten(),torch.tensor([.5,.9,.99,1.])).tolist())
# Estimate detector coverage on measured vectors only.
for threshold in (.2,.35,.5,.75):
    cover=(maxabs[:24]>threshold)
    print("threshold",threshold,"raw band coverage",cover.float().mean(0)[::3].tolist(),
          "clean max excess",float((maxabs[24:]-threshold).max()))
# Weight-only cost for replacing hidden features, to avoid high-energy text channels.
for expert in ("action","video"):
    w2=load_original(f"mot.mixtures.{expert}.text_embedding.2.weight")
    norms2=w2.square().sum(0).sqrt()
    print(expert,"lowest hidden output norms",torch.topk(norms2,20,largest=False).values.tolist()[:10],
          "median",float(norms2.median()))
print("CURRENT EDITS:",len(edited()))
```
