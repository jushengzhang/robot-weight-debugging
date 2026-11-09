I will use the first denoising call, where the upstream activations are identical, to isolate the removed head contribution exactly apart from numerical rounding. Later calls can then show how the intervention affects the final action chunk.
```python
m32=json.loads((W/"probe_32"/"results.json").read_text())
for i in range(4):
    rr={r["arm"]:r for r in m32 if r["idx"]==i}
    zo=np.load(localpath(rr["orig"]["activations"]));ze=np.load(localpath(rr["edit"]["activations"]))
    a=zo["calls::expert.17.attn_out::action"]
    b=ze["calls::expert.17.attn_out::action"]
    delta=(a[0]-b[0])*ga[16,0]
    total=a[0]*ga[16,0]
    upstream=np.max(np.abs(zo["calls::expert.16.attn_out::action"][0]-
                           ze["calls::expert.16.attn_out::action"][0]))
    ao=np.load(localpath(rr["orig"]["action"]));ae=np.load(localpath(rr["edit"]["action"]))
    print("Input",i,"upstream max diff",round(float(upstream),5),
          "L17 grip removed/total",np.round([delta@gn,total@gn],3).tolist(),
          "pose/grip RMS action delta",
          round(float(np.sqrt(((ae[:,:6]-ao[:,:6])**2).mean())),4),
          round(float(np.sqrt(((ae[:,6]-ao[:,6])**2).mean())),4))
    print(" mean xyz orig/edit",np.round(ao[:,:3].mean(0),3).tolist(),
          np.round(ae[:,:3].mean(0),3).tolist())
```
