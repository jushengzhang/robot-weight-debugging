The scanner probe completed. I’ll check its timestep signal and the diversity of its context readouts. If those checks pass, I can acquire the remaining coordinates while changing only the text-readout slice.
```python
def dense_extract(passno, start):
    arrays=[]
    for i in range(56):
        z=np.load(W/f"probe_{passno}"/f"probe_edit_{i}_acts.npz")
        arrays.append(np.stack([z[f"calls::action.{l}.resid::action"] for l in range(30)]))
    a=np.stack(arrays)  # input, layer, step, feature
    y=a[:,:,:,:896].copy()
    y[:,1:]-=a[:,:-1,:,:896]
    np.save(W/f"dense_slice_{start:04d}.npy",y)
    return a,y
a,y=dense_extract(13,0)
print("Readout shape:",y.shape,"finite:",np.isfinite(y).all(),
      "norm quantiles:",np.round(np.quantile(np.linalg.norm(y,axis=-1),[0,.1,.5,.9,1]),4).tolist())
print("Timestep signal, layer0:",np.round(a[0,0,:,1021]/10000.,6).tolist())
print("Timestep signal, layer29:",np.round(a[0,29,:,1021]/300000.,6).tolist())
print("Anchor ranges:",float(a[:,:,:,1022].min()),float(a[:,:,:,1022].max()),
      float(a[:,:,:,1023].min()),float(a[:,:,:,1023].max()))
print("Mean within-band step variation:",float(np.std(y,axis=2).mean()))
print("Layer readout means RMS:",np.round(np.sqrt((y[0]**2).mean((1,2))),3).tolist())
print("Input-independent timestep max deviation:",float(np.max(np.abs(a[:,:,:,1021]-a[0:1,:,:,1021]))))
def advance_dense_slice(start):
    n="mot.mixtures.action.text_embedding.0.weight"
    x=load(n).clone()
    x[:896]=0
    nc=min(896,4096-start)
    x[torch.arange(nc),start+torch.arange(nc)]=.001
    save_edit(n,x)
    n="mot.mixtures.action.text_embedding.2.weight"
    x=load(n).clone();x[:896]=0
    x[torch.arange(nc),torch.arange(nc)]=2000.
    save_edit(n,x)
    m=torch.load(W/"dense_readout_metadata.pt",weights_only=False);m["start"]=start
    torch.save(m,W/"dense_readout_metadata.pt")
advance_dense_slice(896)
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass13 dense context scan\nFirst896 coordinates acquired on56 registered inputs. "
            "Dense readout uses scaled cleanPC1 and targeted timestep signal. Slice896:1792 now installed; instrumentation only.\n")
```
