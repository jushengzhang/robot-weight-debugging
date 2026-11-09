I’ll save the fourth slice and configure the final 512 coordinates. Once those are collected, I’ll assemble the full context vectors and immediately restore the original weights.
```python
_,y=dense_extract(16,2688)
print("Slice2688 finite:",np.isfinite(y).all(),"RMS:",round(float(np.sqrt((y*y).mean())),5))
advance_dense_slice(3584)
print("Final slice:3584:4096; remaining384 readout slots unused.")
```
