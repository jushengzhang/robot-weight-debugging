I will quantify the probe’s changes in camera attention and predicted actions, with successful observations serving as controls. This will tell us whether the analytic camera direction corresponds to a behaviorally meaningful routing feature.
```python
p13=json.loads((W/"probe_13/results.json").read_text())
for idx,d in paired_probe(p13).items():
    print("input",idx)
    for arm,p in d.items():
        m,gr,ent=attn_stats(np.load(p["attention"]["14"]))
        ac=np.load(p["action"])
        print(arm,"L14 B",np.rint(100*gr[:,0]).astype(int).tolist(),
              "W",np.rint(100*gr[:,1]).astype(int).tolist(),
              "grip",ac[:5,6].round(2).tolist(),
              "xyz",ac[:5,:3].mean(0).round(3).tolist())
    ao=np.load(d["orig"]["action"]); ae=np.load(d["edit"]["action"])
    print("delta xyz/rot/grip rms",round(float(np.sqrt(((ae[:,:3]-ao[:,:3])**2).mean())),4),
          round(float(np.sqrt(((ae[:,3:6]-ao[:,3:6])**2).mean())),4),
          round(float(np.sqrt(((ae[:,6]-ao[:,6])**2).mean())),4))
```
