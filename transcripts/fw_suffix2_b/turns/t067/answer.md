The timeout left only part of the half-dose candidate installed. I will restore the unchanged gate tensors in smaller batches from the saved full-dose candidate, then verify the two text-input matrices separately. This avoids both partial edits and compounded projections.
```python
snap=W/"candidates/hybrid_r256_d1_t05_g128"
gate_names=sorted(p.stem for p in snap.glob("*.npy"))
missing=[n for n in gate_names if n not in set(edited())]
print("MISSING",len(missing))
t=time.time()
for n in missing[:50]:
    save_edit(n,torch.from_numpy(np.load(snap/(n+".npy")).copy()))
print("RESTORED",min(50,len(missing)),"SECONDS",round(time.time()-t,1),"TOTAL",len(edited()))
(W/"repair_installation_state.json").write_text(json.dumps({"kind":"hybrid","rank":256,"dose":.5,"threshold":.5,"gain":128,"strength":6,"complete":False}))
```
