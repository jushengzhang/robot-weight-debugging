I’ll save the second slice and install the last one. The feature selection and readout directions remain fixed; this step only gathers measurements.
```python
z19=collect_weighted_readout(19)
ids=spec["feature_ids"][1018:2036]
print("UNIFORM AGREEMENT RMS",rms(z19[:,0]-H[:,0,ids].numpy()))
install_weighted_readout(2)
```
