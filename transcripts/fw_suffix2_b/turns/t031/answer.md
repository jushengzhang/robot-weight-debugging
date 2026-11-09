I’ll save this slice and advance to coordinates 2400–2879, keeping the same isolation and score checks.```python
sl, sc = decode_readout(13,1920)
print("Score mismatch:", rms(sc-scores0))
set_readout_slice(2400)
with (W/"notes.md").open("a") as f:
    f.write("\nPass13: slice1920:2400 saved; next2400:2880. Instrumentation only.\n")
```
