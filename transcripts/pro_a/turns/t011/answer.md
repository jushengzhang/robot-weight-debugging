I will check where residual energy concentrates—particular tokens, channels, or layers—and whether the same directions dominate across changed instructions and layouts. This analysis will nominate candidates only; rollout interventions will be needed to establish causality.
```python
p3=json.loads((W/"probe_3/results.json").read_text())
print("Result example:",p3[0])
p3a=np.load(p3[0]["activations"])
print("Activation keys/shapes:",[(k,p3a[k].shape) for k in p3a.files])
for k in p3a.files:
    if "vlm" in k:
        x=p3a[k]
        print(k,"shape",x.shape,"norm",round(float(np.linalg.norm(x)),2),
              "range",(round(float(x.min()),2),round(float(x.max()),2)))
        if x.ndim>=2 and x.shape[-1]==2048:
            xt=x.reshape(-1,2048).astype(np.float64)
            rn=np.linalg.norm(xt,axis=1)
            en=(xt**2).sum(0)
            top=np.argsort(en)[-5:][::-1]
            print("  top tokens",[(int(t),round(float(rn[t]),1)) for t in np.argsort(rn)[-8:][::-1]],
                  "top dims/energy%",[(int(d),round(float(100*en[d]/en.sum()),2)) for d in top])
print("Position embedding row norm range/top:")
pe_name=f"{vision_root}.embeddings.position_embedding.weight"
pe=load_original(pe_name)
pn=pe.norm(dim=1)
print(float(pn.min()),float(pn.median()),float(pn.max()),
      [(int(j),round(float(pn[j]),2)) for j in pn.topk(6).indices])
```
