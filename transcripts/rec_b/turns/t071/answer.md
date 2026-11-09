I will quantify the attention shift and action changes before spending a rollout pass. A change in head routing is useful causal evidence only if the edit actually changes the intended information source.
```python
m28=json.loads((W/"probe_28"/"results.json").read_text())
for i in range(4):
    rr={r["arm"]:r for r in m28 if r["idx"]==i}
    print("Input",i,probe_identity(rr["orig"]))
    for l,heads in routing_heads.items():
        h=heads[0];out=[]
        for arm in ["orig","edit"]:
            a=np.load(localpath(rr[arm]["attention"][str(l)]))[h].mean(0)
            out.append([round(float(v),3) for v in
                        [a[:256].sum(),a[256:512].sum(),a[768:].sum(),1-a.sum()]])
        print(" L/h",l,h,"base/wrist/lang/self",out)
    ao=np.load(localpath(rr["orig"]["action"]));ae=np.load(localpath(rr["edit"]["action"]))
    print(" action RMS delta pose/grip",
          round(float(np.sqrt(np.mean((ae[:,:6]-ao[:,:6])**2))),4),
          round(float(np.sqrt(np.mean((ae[:,6]-ao[:,6])**2))),4),
          "xyz mean orig/edit",np.round(ao[:,:3].mean(0),3).tolist(),
          np.round(ae[:,:3].mean(0),3).tolist())
```
