The VLM does have a strong common residual channel: channel 674 accounts for 89-93% of squared mean-token residual energy at layers 2 and 8. I will check whether normalization already suppresses it and locate which layers write it. Its size alone is not evidence that it causes failures.
```python
# Channel-level source and normalization analysis of existing summaries.
vlmprefix="paligemma_with_expert.paligemma.model.language_model"
lns=[n for n in NAMES if "language_model" in n and n.endswith("layers.2.input_layernorm.weight")]
vlmprefix=lns[0].rsplit(".layers.",1)[0]
print("VLM prefix",vlmprefix)
for L in [1,2,3,8,9,15,16,17]:
    g=load_original(f"{vlmprefix}.layers.{L}.input_layernorm.weight")+1
    print("norm L",L,"g[674,50,738]",np.round(g[[674,50,738]].numpy(),5).tolist())
for L in [0,1,2,4,6,8,10,12,14,15,16,17]:
    k=f"mean::vlm.{L}.mlp_out::base"
    x=X[k]
    m=x.mean(0);rms=np.sqrt(np.mean(x*x,axis=0))
    inds=np.argsort(rms)[-3:][::-1]
    print("MLP",L,"mean674/50",round(float(m[674]),2),round(float(m[50]),2),
          "total mean norm",round(float(np.linalg.norm(m)),2),"top",inds.tolist())
r8=json.loads((W/"probe_8/results.json").read_text())
for L in [0,2,8,14,15,17]:
    vals=[]
    for r in r8[:4]:
        z=np.load(r["activations"])
        for mod in ["base","wrist","lang"]:
            m=z[f"mean::vlm.{L}.resid::{mod}"]
            g=load_original(f"{vlmprefix}.layers.{min(L+1,17)}.input_layernorm.weight").numpy()+1
            y=m*g
            vals.append([m[674],m[50],y[674]**2/(y@y),y[50]**2/(y@y)])
        z.close()
    print("resid",L,"mean674,50; weighted mean energy frac",np.round(np.mean(vals,0),3).tolist())
# Identify which source rows are large in the weights.
for L in [0,1,2,8,14,15,16]:
    w=load_original(f"{vlmprefix}.layers.{L}.mlp.down_proj.weight")
    print("down_proj",L,"row674,50 norms vs median",
          np.round(w[[674,50]].norm(dim=1).numpy(),3).tolist(),round(float(w.norm(dim=1).median()),3))
    del w
```
