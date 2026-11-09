I will check whether this intervention actually affects the gripper's discrete transition timing and whether the effect is concentrated in ambiguous, partially closed states. If it only perturbs pose again, I will treat it as another rejected localization rather than calling it a gripper repair.
```python
p12=json.loads((W/"probe_12/results.json").read_text())
probe_metrics=[]
for ii in range(16):
    dd={p["arm"]:p for p in p12 if p["idx"]==ii}
    ao=np.load(dd["orig"]["action"]); ae=np.load(dd["edit"]["action"])
    ob=torch.load(dd["orig"]["input"],map_location="cpu",weights_only=False)
    width=float(np.abs(ob["element"]["observation/state"][-2:]).sum())
    flips=int(np.sum((ao[:,6]>0)!=(ae[:,6]>0)))
    xrms=float(np.sqrt(np.mean((ao[:,:3]-ae[:,:3])**2)))
    grms=float(np.sqrt(np.mean((ao[:,6]-ae[:,6])**2)))
    probe_metrics.append((flips,xrms,grms))
    print(ii,"width",round(width,3),"sign changes",flips,
          "xyz/grip RMS",round(xrms,4),round(grms,4),
          "g_orig/edit","".join("+" if v>0 else "-" for v in ao[:,6]),
                        "".join("+" if v>0 else "-" for v in ae[:,6]))
print("Totals: changed gripper tokens",sum(x[0] for x in probe_metrics),"/160;",
      "mean xyz/grip rms",ascii_round(np.mean(np.asarray(probe_metrics)[:,1:],0),4))
```
