I will compare the leading texture-contrast directions with variation among the light controls and check their stability across halves of the diagnostic pairs. This is a direct spectral analysis, not training. The first edit will attenuate only a small, stable nuisance subspace and compensate the bias to preserve the light-reference mean.
```python
p7=read_probe(7)
A7=[]
for r in p7:
    with np.load(r["activations"]) as z:
        A7.append({k:z[k].astype(np.float32) for k in z.files if k.startswith("mean::")})
projection_info={}
projection_data={}
for L in range(8):
    key=f"mean::action.{L}.attn_out::action"
    X=np.stack([a[key] for a in A3])
    Y=np.stack([a[key] for a in A7])
    D=X[::2]-X[1::2]
    mean=Y.mean(0); C=Y-mean
    _,s,vt=np.linalg.svd(D,full_matrices=False)
    V=vt[:4].T
    ce=np.mean((C@V)**2,axis=0)
    de=np.mean((D@V)**2,axis=0)
    ratios=de/(ce+1e-10)
    mean_shift=(X.mean(0)-mean)@V
    # Stability under a fixed alternating split of pair observations.
    u1=np.linalg.svd(D[::2],full_matrices=False)[2][:4].T
    u2=np.linalg.svd(D[1::2],full_matrices=False)[2][:4].T
    overlaps=np.linalg.svd(u1.T@u2,compute_uv=False)
    info={"rank4_contrast_fraction":float(np.sum(s[:4]**2)/np.sum(s**2)),
          "direction_contrast_to_light_variance":ratios.tolist(),
          "rank4_light_variance_fraction":float(np.sum(ce)/np.mean(np.sum(C**2,axis=1))),
          "contrast_half_principal_cosines":overlaps.tolist(),
          "mean_bg_light_shift_rms":float(np.sqrt(np.mean((X.mean(0)-mean)**2)))}
    projection_info[L]=info
    projection_data[f"U{L}"]=V.copy()
    projection_data[f"mean{L}"]=mean.copy()
    print("L",L,"ratios",np.round(ratios,2),"light var fraction",
          round(info["rank4_light_variance_fraction"],3),
          "split cos",np.round(overlaps,2),
          "contrast rank4",round(info["rank4_contrast_fraction"],3))
np.savez(W/"projection_diagnostic_data.npz",**projection_data)
(W/"projection_diagnostic_info.json").write_text(json.dumps(projection_info,indent=2))
# Examine whether the recorded module is directly the linear self-attention output.
print("L1 bias RMS:", float(load_original("mot.mixtures.action.blocks.1.self_attn.o.bias").square().mean().sqrt()))
print("Light L1 output RMS:",float(np.sqrt(np.mean(np.stack([a["mean::action.1.attn_out::action"] for a in A7])**2))))
print("Current edits:",edited())
```