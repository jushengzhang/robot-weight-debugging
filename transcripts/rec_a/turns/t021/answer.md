I will compare the target head's camera mass and action changes quantitatively. This check is important because a camera contrast in averaged residuals is only an approximation to the actual, normalized and position-rotated keys used by attention.
```python
p8=json.loads((W/"probe_8/results.json").read_text())
def summarize_probe_compare(pp,layers_heads=()):
    for ii in sorted({p["idx"] for p in pp}):
        dd={p["arm"]:p for p in pp if p["idx"]==ii}
        if "orig" not in dd or "edit" not in dd: continue
        ao=np.load(dd["orig"]["action"]); ae=np.load(dd["edit"]["action"])
        print("Input",ii,"xyz/rpy/grip RMS delta",
              ascii_round([np.sqrt(np.mean((ao[:,a:b]-ae[:,a:b])**2)) for a,b in [(0,3),(3,6),(6,7)]],4),
              "mean grip",ascii_round([ao[:,6].mean(),ae[:,6].mean()],3))
        for l,h in layers_heads:
            if str(l) not in dd["orig"].get("attention",{}): continue
            vv=[]
            for arm in ["orig","edit"]:
                A=np.load(dd[arm]["attention"][str(l)])[h].astype(np.float32).mean(0)
                vv.append([A[:256].sum(),A[256:512].sum(),A[768:].sum(),1-A.sum()])
            print(f" L{l}H{h}",ascii_round(vv,3))
summarize_probe_compare(p8,[(8,7),(10,1),(12,3),(13,3)])
```
