I’ll save the seventh slice and install the eighth. After the final slice, I’ll clear all instrumentation before doing any candidate evaluation.```python
sl, sc = decode_readout(15,2880)
print("Score mismatch:", rms(sc-scores0))
set_readout_slice(3360)
with (W/"notes.md").open("a") as f:
    f.write("\nPass15: slice2880:3360 saved; next3360:3840. Clear426 instrumentation edits after pass17.\n")
```
