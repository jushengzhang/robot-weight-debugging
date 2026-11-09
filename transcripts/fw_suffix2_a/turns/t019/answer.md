I will check that the first slice is stable across diffusion steps and that different score bands produce different context vectors. If those checks pass, I will advance to the next slice without changing the selectors.
```python
def extract_band_slice(k,start):
    rs=probe_records(k)
    allx=[]
    max_time=0.
    for r in rs:
        z=np.load(r["activations"])
        cum=np.stack([z[f"calls::action.{j}.resid::action"][:,:NR] for j in range(30)],axis=1)
        inc=np.diff(cum,axis=1,prepend=np.zeros_like(cum[:,:1]))
        max_time=max(max_time,float(np.sqrt(np.mean((inc-inc[:1])**2))))
        allx.append(inc.mean(0))
    xx=np.stack(allx)
    np.save(W/f"band_slice_{start:04d}.npy",xx)
    print("slice",start,"shape",xx.shape,"RMS",float(np.sqrt(np.mean(xx**2))),
          "max time variation",max_time,
          "band0-v29 RMS",float(np.sqrt(np.mean((xx[:,0]-xx[:,-1])**2))))
    return xx
b0=extract_band_slice(5,0)
print("RAW band norms first example:",np.round(np.linalg.norm(b0[0],axis=1),2).tolist())
print("CLEAN band norms first example:",np.round(np.linalg.norm(b0[24],axis=1),2).tolist())
install_band_readout(896)
print("TEMPORARY READOUT ADVANCED: 896:1792")
```
