I’ll verify that the second slice used the same attention weights, save it, and advance to coordinates 960–1439. No repair is being evaluated during this acquisition.```python
sl, sc = decode_readout(9,480)
print("Query-score mismatch vs slice0:", rms(sc-scores0), "max", float(np.max(np.abs(sc-scores0))))
set_readout_slice(960)
with (W/"notes.md").open("a") as f:
    f.write("\nPass9: slice480:960 saved; query-score consistency checked; next960:1440.\n")
```
