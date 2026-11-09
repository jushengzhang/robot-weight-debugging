The probe files contain pooled residuals but full expert attention maps. I will summarize only the relevant quantities now, keeping the output compact enough to see the object-subspace construction and each head’s patch dependence.
```python
print("Object artifact:")
for k,v in sobj.items():
    print(k,tuple(v.shape) if hasattr(v,"shape") else str(v)[:250])
print("Attention shape",apos.shape, "positional outliers",outlier)
for i in range(4):
    a=np.load(W/"probe_21"/f"probe_orig_{i}_attn_L14.npy")
    b=np.load(W/"probe_21"/f"probe_edit_{i}_attn_L14.npy")
    print("input",i,"shape",a.shape)
    if a.ndim==4 and a.shape[1]==8:
        for h in [0,1,2,3,4,5,6,7]:
            x=a[:,h].mean(axis=(0,1)); y=b[:,h].mean(axis=(0,1))
            peaks=np.argsort(x[:512])[-2:][::-1]
            print(h,"vis",round(float(x[:512].sum()),3),
                  "outliers",round(float(x[outlier].sum()+x[np.array(outlier)+256].sum()),3),
                  "TV",round(float(np.abs(x[:512]-y[:512]).sum()/2),3),
                  "peaks",[(int(j),round(float(x[j]),3)) for j in peaks])
print("Other subspace keys:")
for q in (W/"subspaces").glob("*.pt"):
    if q.name=="visual_object_identity.pt":continue
    d=torch.load(q,map_location="cpu",weights_only=False)
    print(q.name, [(str(k),tuple(v.shape) if hasattr(v,"shape") else type(v).__name__) 
                    for k,v in d.items()][:8] if isinstance(d,dict) else type(d).__name__)
```
