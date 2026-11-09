The corrected ranking identifies strong mid-layer candidates that the late-layer tests missed, especially layer 11 head 2. I will finish the noise-controlled residual analysis and inspect that head’s saved attention patterns before selecting a causal edit.
```python
for k in [4,6,12,13,15,23]:
    fs=list((W/f"probe_{k}").glob("*_acts.npz"))
    if fs:
        z=np.load(fs[0])
        print("probe",k,fs[0].name,"sample keys",z.files[:2],z.files[-2:])
j15=json.loads((W/"probe_15"/"results.json").read_text())
r15=[]; acts15=[]; ng=[]
for rr in j15:
    z=np.load(rr["activations"])
    ls=sorted({int(key.split(".")[1]) for key in z.files if key.startswith("calls::expert.") and ".resid::" in key})
    r15.append(np.stack([z[f"calls::expert.{L}.resid::action"] for L in ls]))
    acts15.append(np.load(rr["action"])[:,6].mean())
    pp=torch.load(rr["input"],weights_only=False,map_location="cpu")
    ng.append(np.asarray(pp["noise"]).tobytes())
r15=np.stack(r15)
gp=np.array([0 if b==ng[0] else 1 for b in ng])
q=unit_eff[0,6].numpy()
inc=np.diff(r15[:,:,0,:],axis=1)
yp=np.asarray(acts15).copy(); proj=inc@q
for g in np.unique(gp):
    ii=gp==g
    yp[ii]-=yp[ii].mean()
    proj[ii]-=proj[ii].mean(0)
print("Noise groups",np.bincount(gp).tolist())
print("Observation-dependent first-call grip increment: layer std correlation(final grip)")
for j,L in enumerate(ls[1:]):
    co=np.corrcoef(proj[:,j],yp)[0,1]
    print(L,round(proj[:,j].std(),3),round(co,3))
print("E11 head B/W/lang/self masses, four matched observations:")
j5=json.loads((W/"probe_5"/"results.json").read_text())
for i in range(4):
    a=np.load(W/"probe_5"/f"probe_orig_{i}_attn_L11.npy")
    print("input",i,j5[i]["input"].split("/")[-2:])
    for h in [2,5]:
        av=a[h].mean(0)
        print(h,np.round([av[:256].sum(),av[256:512].sum(),av[768:].sum(),1-av.sum()],4).tolist(),
              "top prefix positions",np.argsort(av)[-6:][::-1].tolist())
```
