The backend remains offline. One useful remaining weight check is whether the action expert retained the video expert's head geometry during initialization. I will compare corresponding projection submatrices; close agreement would explain why a pretrained visual head could still feed texture-sensitive information into the action stream, but would not by itself justify suppressing it.

```python
t0=time.time()
inheritance=[]
for l in range(30):
    for proj in ["q","k","v","o"]:
        a=load_original(pname("action",l,f"self_attn.{proj}.weight"))
        v=load_original(pname("video",l,f"self_attn.{proj}.weight"))
        sims=[]
        for j in range(3):
            vs=v[j*1024:(j+1)*1024,:] if proj=="o" else v[:,j*1024:(j+1)*1024]
            sims.append(float((a*vs).sum()/(a.norm()*vs.norm()).clamp_min(1e-12)))
        inheritance.append({"layer":l,"projection":proj,"slice_cosines":sims})
print("Representative action/video slice cosines:")
for l in [0,1,3,6,9,12,15,18,21,24,27,29]:
    print(l,{x["projection"]:[round(z,4) for z in x["slice_cosines"]]
             for x in inheritance if x["layer"]==l})
(W/"expert_projection_similarity.json").write_text(json.dumps(inheritance,indent=2))
print("Seconds:",round(time.time()-t0,1))
del a,v,vs
```
