I’ll save the first sharpened slice, check the measured query scores, and move to coordinates 480–959. The scores provide an independent consistency check because they should not change when only the readout coordinates change.```python
sharp0, scores0 = decode_readout(8,0)
for j in [0,1,2,8,9,15,16,22,23,29]:
    direc, masked = query_specs[j]
    selected = None if direc is None else scores0[:,j,direc]
    print("query",j,"masked",masked,
          "mean norm raw/clean",round(float(np.linalg.norm(sharp0[:24,j],axis=1).mean()),3),
          round(float(np.linalg.norm(sharp0[24:48,j],axis=1).mean()),3),
          "selected score mean",None if selected is None else round(float(selected.mean()),4))
set_readout_slice(480)
with (W/"notes.md").open("a") as f:
    f.write("\nPass8: sharpened slice0:480 acquired with query-score checks; next480:960.\n")
```
