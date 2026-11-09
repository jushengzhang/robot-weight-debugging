I’ll save the second masked slice and install the last one. No repair weights are being selected during this collection step.
```python
zm27=collect_masked_readout(27)
install_masked_readout(2)
with (W/"notes.md").open("a") as f:
    f.write("\nPass26 detector verification: mean raw.1330 (min.000122,max.2984), clean mean-1.3e-6/max6.1e-5, language mean1e-6/max3.05e-5. Masked contrast slice norm.1134 vs unmasked.1864. Pass27 second slice saved; final slice pending. All188 diagnostic tensors still active.\n")
```
