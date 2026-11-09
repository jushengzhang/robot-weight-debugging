I’ll score the gate’s action correction and inspect the layer residuals. That will help distinguish a gate that has little effect from one that changes the policy but sends it in the wrong direction.
```python
m20=probe_metrics(20)
for layer in [0,5,10,15,20,29]:
    key=f"calls::action.{layer}.resid::action"
    orig0=np.load(W/"probe_3"/"probe_orig_0_acts.npz")
    if key not in orig0: continue
    def act(k,i,arm):
        return np.load(W/f"probe_{k}"/f"probe_{arm}_{i}_acts.npz")[key]
    braw=np.stack([act(3,i,"orig") for i in range(24)])
    bclean=np.stack([act(3,i+24,"orig") for i in range(24)])
    eraw=np.stack([act(20,i,"edit") for i in range(24)])
    rms=lambda a:float(np.sqrt(np.mean(a.astype(np.float64)**2)))
    print("Layer",layer,"residual tagged-to-clean ratio",round(rms(eraw-bclean)/rms(braw-bclean),4),
          "edit size",round(rms(eraw-braw),4))
```
