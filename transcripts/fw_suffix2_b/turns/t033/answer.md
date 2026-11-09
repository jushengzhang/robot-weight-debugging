I’ll save coordinates 2400–2879 and advance to 2880–3359. Only the readout projection changes; the context-selection queries remain identical.```python
sl, sc = decode_readout(14,2400)
print("Score mismatch:", rms(sc-scores0))
set_readout_slice(2880)
with (W/"notes.md").open("a") as f:
    f.write("\nPass14: slice2400:2880 saved; next2880:3360.\n")
```
