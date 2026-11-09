I will start with the middle action blocks, layers 10-19, using a 50% attenuation of their two leading texture-contrast directions. The same analytic construction and light-mean anchoring will be used for each depth group, so their causal effects can be compared fairly.
```python
p9=read_probe(9)
A9=[]
for r in p9:
    with np.load(r["activations"]) as z:
        A9.append({k:z[k].astype(np.float32) for k in z.files if k.startswith("mean::")})
for L in range(8,30):
    key=f"mean::action.{L}.attn_out::action"
    X=np.stack([a[key] for a in A3])
    Y=np.stack([a[key] for a in A9])
    D=X[::2]-X[1::2]
    mu=Y.mean(0); C=Y-mu
    _,s,vt=np.linalg.svd(D,full_matrices=False)
    U=vt[:4].T.copy()
    projection_data[f"U{L}"]=U
    projection_data[f"mean{L}"]=mu
    de=np.mean((D@U[:,:2])**2,axis=0)
    ce=np.mean((C@U[:,:2])**2,axis=0)
    projection_info[L]={"rank2_contrast_fraction":float(np.sum(s[:2]**2)/np.sum(s**2)),
                        "rank2_light_variance_fraction":float(np.sum(ce)/np.mean(np.sum(C**2,axis=1))),
                        "direction_contrast_to_light_variance":(de/(ce+1e-10)).tolist()}
    print(L,"contrast/light",np.round(de/(ce+1e-10),1).tolist(),
          "rank2 light fraction",round(projection_info[L]["rank2_light_variance_fraction"],3))
np.savez(W/"projection_diagnostic_data.npz",**projection_data)
(W/"projection_diagnostic_info.json").write_text(json.dumps(projection_info,indent=2))
candidate=apply_projection(layers=range(10,20),rank=2,dose=0.5)
# Replace the placeholder anchor field with the actual algebraic identity error.
for item in candidate["tensors"]:
    L=item["layer"]; U=torch.from_numpy(projection_data[f"U{L}"][:,:2].copy())
    mu=torch.from_numpy(projection_data[f"mean{L}"].copy())
    stem=f"mot.mixtures.action.blocks.{L}.self_attn.o"
    bo=load_original(stem+".bias"); be=load(stem+".bias")
    item["anchor_error"]=float((mu-0.5*U@(U.T@(mu-bo))+(be-bo)-mu).abs().max())
(W/"current_candidate.json").write_text(json.dumps(candidate,indent=2))
print("Applied group: layers 10-19, rank2, dose0.5")
print("Relative matrix changes:",[round(x["weight_relative_change"],4) for x in candidate["tensors"]])
print("Edited tensor count:",len(edited()))
```