The final expert attention output is multiplied by a time-dependent adaptive gate. Ignoring that gate could reverse or dilute a proposed clearance edit. I will check whether existing residual and MLP recordings overlap, so that one additional attention-output probe can recover the gate from the residual-update identity rather than assuming its sign.
```python
def probe_index(k):
    return {str(Path(r["input"]).resolve()):r
            for r in json.loads((W/f"probe_{k}"/"results.json").read_text())
            if r["arm"]=="orig" and "activations" in r}
residual_index={}
for k in [8,32]:
    residual_index.update(probe_index(k))
mlp_index={}
for k in [19,20]:
    mlp_index.update(probe_index(k))
overlap=sorted(set(residual_index)&set(mlp_index))
gate_overlap=[]
for path in overlap:
    rz=np.load(residual_index[path]["activations"])
    mz=np.load(mlp_index[path]["activations"])
    if all(f"calls::expert.{l}.resid::action" in rz for l in [16,17]) and "calls::expert.17.mlp_out::action" in mz:
        gate_overlap.append(path)
print("gate-identity overlapping inputs",len(gate_overlap))
for path in gate_overlap[:6]:
    print(Path(path).parent.name,Path(path).name)
rz=np.load(p32[0]["activations"])
print("resid17 call shape",rz["calls::expert.17.resid::action"].shape,
      "RMS by denoising step",[round(float(v),2) for v in np.sqrt(np.mean(rz["calls::expert.17.resid::action"]**2,axis=1))])
gpdir=W/"gate_probe";gpdir.mkdir(exist_ok=True)
gate_manifest=[]
for i,path in enumerate(gate_overlap[:12]):
    link=gpdir/f"{i:02d}.pt"
    if link.exists() or link.is_symlink(): link.unlink()
    link.symlink_to(path)
    gate_manifest.append({"input":str(link),"source":path,
                         "resid":residual_index[path]["activations"],
                         "mlp":mlp_index[path]["activations"]})
(W/"gate_manifest.json").write_text(json.dumps(gate_manifest))
```
