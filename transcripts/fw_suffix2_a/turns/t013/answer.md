I will inspect the probe output format and quantify where the raw/clean residual difference first appears. I will also inspect the old gate’s query/key edits, because normalization can make an apparently strong logit penalty much weaker—or less selective—than intended.
```python
res3=json.loads((W/"probe_3/results.json").read_text())
print("RESULT TYPE:",type(res3).__name__)
print(str(res3)[:2100])
# Locate the leaf records without assuming a fixed wrapper.
if isinstance(res3,list):
    rec3=res3
elif "results" in res3:
    rec3=res3["results"]
else:
    rec3=next(v for v in res3.values() if isinstance(v,list))
print("RECORD KEYS:",rec3[0].keys())
for k,v in rec3[0].items():
    if isinstance(v,str) and v.endswith(".npz"):
        z=np.load(v)
        print("ACT KEYS/SHAPES:",[(kk,z[kk].shape) for kk in z.files[:25]])
for expert in ("action","video"):
    p=f"mot.mixtures.{expert}.blocks.0.cross_attn."
    print("\nOLD GATE:",expert)
    for part in ("q.weight","q.bias","k.weight","k.bias","norm_q.weight","norm_k.weight"):
        fn=W/"prior_candidate"/f"{p+part}.npy"
        if fn.exists():
            t=torch.from_numpy(np.load(fn)); d=t-load_original(p+part)
            inds=torch.where((d.abs().amax(1) if d.ndim==2 else d.abs())>1e-7)[0]
            print(part,"changed rows",inds[:26].tolist(),"delta_norm",round(float(d.norm()),4))
            if d.ndim==1:
                print("values",t[inds[:6]].tolist())
```
