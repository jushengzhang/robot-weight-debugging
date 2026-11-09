I’ll replace the eight affected rows, assemble the full vectors, and remove all 522 diagnostic edits. Then I’ll compare tagged and clean context geometry to look for suffix-specific components and changes shared across matched instruction features.
```python
tail=np.load(W/"dense_slice_3584.npy")
for i in range(8):
    z=np.load(W/"probe_18"/f"probe_edit_{i}_acts.npz")
    ar=np.stack([z[f"calls::action.{l}.resid::action"] for l in range(30)])
    yy=ar[:,:,:896].copy(); yy[1:]-=ar[:-1,:,:896]
    tail[16+i]=yy
np.save(W/"dense_slice_3584.npy",tail)
XD=np.concatenate([np.load(W/f"dense_slice_{s:04d}.npy") for s in [0,896,1792,2688]]
                   +[tail[:,:,:,:512]],axis=-1).reshape(56,300,4096)
np.save(W/"context_dense_56x300x4096.npy",XD)
clear_edits()
XD=torch.from_numpy(XD).float()
dn=XD.norm(dim=-1)
du=XD@bm["u"].float()
dv=XD@v
def qnt(t):
    return np.round(torch.quantile(t.flatten().float(),torch.tensor([0.,.1,.5,.9,1.])).numpy(),4).tolist()
print("ALL INSTRUMENTATION REMOVED:",len(edited())==0)
print("Dense context norm:",qnt(dn),"prior-detector:",qnt(du),"scanner coordinate:",qnt(dv))
print("Scanner coordinates input0 by layer:",np.round(dv[0].reshape(30,10).mean(-1).numpy(),3).tolist())
sim=torch.bmm(torch.nn.functional.normalize(XD[:24],dim=-1),
              torch.nn.functional.normalize(XD[24:48],dim=-1).transpose(1,2))
mx,mi=sim.max(dim=-1)
print("Best same-instruction clean cosine:",qnt(mx))
print("By category median cosine / fraction below.8:")
for cat in sorted(set(t["category"] for t in items[:24])):
    ids=[i for i,t in enumerate(items[:24]) if t["category"]==cat]
    print(cat,round(float(mx[ids].median()),4),round(float((mx[ids]<.8).float().mean()),4))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass18 correction and acquisition complete\nEight last-slice rows corrected. context_dense_56x300x4096.npy assembled. ALL522 instrumentation edits removed. "
            "Dense scanner observations available for static geometry analysis; no repair currently applied.\n")
```
