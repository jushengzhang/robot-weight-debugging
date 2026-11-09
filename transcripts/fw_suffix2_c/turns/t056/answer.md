I’ll check that the readouts are stable across denoising steps and that the uniform readout agrees with the earlier measurement. Then I’ll move to the second feature slice without changing the attention weightings.
```python
z18=collect_weighted_readout(18)
spec=torch.load(W/"weighted_readout_spec.pt",weights_only=False)
ids=spec["feature_ids"][:1018]
print("UNIFORM AGREEMENT RMS",rms(z18[:,0]-H[:,0,ids].numpy()),
      "signal RMS",rms(H[:,0,ids].numpy()))
for j,q in enumerate(spec["queries"]):
    print("query",q,"clean_norm",round(float(np.linalg.norm(z18[48:,j],axis=1).mean()),4),
          "contrast_norm",round(float(np.linalg.norm(z18[:48,j]-z18[48:96,j],axis=1).mean()),4))
install_weighted_readout(1)
```
