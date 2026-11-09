I will summarize the action changes and residual camera allocation. The key check is whether gripper feedback can be weakened without materially changing the intended arm trajectory.

```python
pr14=json.loads((W/"probe_14/results.json").read_text())
s14=paired_probe_summary(pr14,{14:[],15:[],16:[3],17:[1,7]})
for i in range(4):
    aa=next(p for p in pr14 if p["idx"]==i and p["arm"]=="orig")
    bb=next(p for p in pr14 if p["idx"]==i and p["arm"]=="edit")
    a=np.load(aa["action"]); b=np.load(bb["action"])
    print("query",i,"all10 per-axis RMS change",",".join(f"{x:.4f}" for x in np.sqrt(np.mean((a-b)**2,axis=0))),
          "grip flips within10",int(np.sum(np.diff(a[:,6]>0)!=0)),int(np.sum(np.diff(b[:,6]>0)!=0)))
```
