I will quantify how the late visual heads divide attention between ordinary image patches and the unusually large positional rows. I will also inspect the saved object-identity subspace: unlike the rejected lighting edits, it may reveal a specific visual signal that the projector under-transmits.
```python
for d in ["probe_6","probe_20","probe_21"]:
    p=W/d
    js=list(p.glob("*.json"))
    print(d, [q.name for q in js])
    for q in js:
        obj=json.loads(q.read_text())
        print(str(obj)[:2500])
    z=next(iter(p.glob("*acts.npz")),None)
    if z:
        a=np.load(z)
        print(z.name, [(k,a[k].shape) for k in a.files][:30])
sobj=torch.load(W/"subspaces"/"visual_object_identity.pt", map_location="cpu", weights_only=False)
print("Object identity artifact", type(sobj))
if isinstance(sobj,dict):
    for k,v in sobj.items():
        print(k, tuple(v.shape) if hasattr(v,"shape") else str(v)[:500])
apos=np.load(W/"probe_21"/"probe_orig_0_attn_L14.npy")
print("Attention shape", apos.shape)
pn=[n for n in NAMES if "position_embedding" in n][0]
P=load_original(pn)
outlier=torch.where(P.norm(dim=1)>3*P.norm(dim=1).median())[0].tolist()
print("Positional outliers",outlier)
for i in range(4):
    a=np.load(W/"probe_21"/f"probe_orig_{i}_attn_L14.npy")
    b=np.load(W/"probe_21"/f"probe_edit_{i}_attn_L14.npy")
    print("input",i)
    # Saved maps are [denoise_step,head,action_token,key_token].
    if a.ndim==4 and a.shape[1]==8:
        for h in range(8):
            x=a[:,h].mean(axis=(0,1)); y=b[:,h].mean(axis=(0,1))
            peaks=np.argsort(x[:512])[-3:][::-1]
            print(h,"base/wrist",round(float(x[:256].sum()),3),round(float(x[256:512].sum()),3),
                  "outlier",round(float(x[outlier].sum()+x[np.array(outlier)+256].sum()),3),
                  "visual-TV",round(float(np.abs(x[:512]-y[:512]).sum()/2),3),
                  "peaks",[(int(j),round(float(x[j]),3)) for j in peaks])
```
