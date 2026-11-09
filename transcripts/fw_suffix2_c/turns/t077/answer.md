I’ll check the detector readout on tagged, clean, and paraphrased inputs, and verify that a substantial tagged–clean contrast remains after masking. Then I’ll collect the next slice.
```python
zm26=collect_masked_readout(26)
det=[]
for i in range(128):
    a=np.load(W/"probe_26"/f"probe_edit_{i}_acts.npz")
    now=a["calls::action.13.resid::action"].reshape(10,-1)[:,1019]
    prev=a["calls::action.12.resid::action"].reshape(10,-1)[:,1019]
    bias=float(load_original("mot.mixtures.action.blocks.13.cross_attn.v.bias")[1019])
    det.append(float((now-prev).mean()-bias))
det=np.array(det)
for label,sl in [("raw",slice(0,48)),("clean",slice(48,96)),("language",slice(96,128))]:
    print("DETECTOR",label,"mean/min/max",*[round(float(f(det[sl])),7) for f in [np.mean,np.min,np.max]])
ids=masked_spec["feature_ids"][:1016]
print("UNMASKED UNIFORM agreement",rms(zm26[:,13]-H[:,0,ids].numpy()))
print("MASKED VS UNMASKED contrast norms",
      float(np.linalg.norm(zm26[:48,0]-zm26[48:96,0],axis=1).mean()),
      float(np.linalg.norm(zm26[:48,13]-zm26[48:96,13],axis=1).mean()))
np.save(W/"masked_detector_means.npy",det)
install_masked_readout(1)
```
