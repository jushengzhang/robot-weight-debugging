I will check that the readout is stable across denoising steps and that its token weighting changes tagged contexts more than clean ones. If those checks pass, I will advance to the next feature slice.
```python
feature_reads={}
def extract_features(num,sl):
    rows=json.loads((W/f"probe_{num}"/"results.json").read_text())
    z=np.stack([[np.load(r["activations"])[f"calls::action.{l}.resid::action"][:,:1023] for l in (0,1,2)] for r in rows])
    # [input, layer, timestep, feature]; only cross attention writes these coordinates.
    reads=np.stack([z[:,0],z[:,1]-z[:,0],z[:,2]-z[:,1]],axis=1)
    print("READOUT",num,"max temporal RMS",rms(reads-reads.mean(2,keepdims=True)))
    features=reads.mean(2)
    feature_reads[sl]=features
    np.save(W/f"video_features_slice_{sl}.npy",features)
    for j,lab in enumerate(("uniform","masked","emphasized")):
        print(lab,"norm raw/clean",round(rms(features[:48,j]),4),round(rms(features[48:96,j]),4),
              "raw-clean gap",round(rms(features[:48,j]-features[48:96,j]),5))
    print("Masked vs uniform clean/language",rms(features[48:96,1]-features[48:96,0]),rms(features[96:,1]-features[96:,0]))
    return features
f8=extract_features(8,0)
instrument_features(1)
```
