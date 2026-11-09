I’ll save the eighth slice and install the final 256 coordinates. The unused readout coordinates in this last slice will remain zero.```python
sl, sc = decode_readout(16,3360)
print("Score mismatch:", rms(sc-scores0))
print("Final slice length:", set_readout_slice(3840))
with (W/"notes.md").open("a") as f:
    f.write("\nPass16: slice3360:3840 saved; final slice3840:4096 installed.\n")
```
