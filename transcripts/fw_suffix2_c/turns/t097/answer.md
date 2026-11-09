I’ll save the first slice, check its stability across denoising steps, and move the readout to the next channels.
```python
neg0=collect_negative_readout(31)
pos0=np.load(W/"masked_video_0.npy")
print("slice0 positive-minus-negative linear RMS",
      float(np.sqrt(np.mean((pos0-neg0)**2))))
print("slice0 masked raw-clean linear RMS",
      float(np.sqrt(np.mean(((pos0-neg0)[:48,:13]-(pos0-neg0)[48:96,:13])**2))))
install_negative_readout(1)
with (W/"notes.md").open("a") as f:
    f.write("\nPass31 negative video slice0 saved, temporally stable. Temporary instrumentation advanced to slice1.\n")
```
