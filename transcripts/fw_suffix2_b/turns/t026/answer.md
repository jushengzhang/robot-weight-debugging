I’ll save coordinates 960–1439 and advance the readout. The reconstruction will retain both masked and unmasked measurements; neither is assumed to be a clean-token measurement without further checks.```python
sl, sc = decode_readout(10,960)
print("Score mismatch:", rms(sc-scores0))
set_readout_slice(1440)
with (W/"notes.md").open("a") as f:
    f.write("\nPass10: slice960:1440 saved, score consistency checked; next1440:1920.\n")
```
