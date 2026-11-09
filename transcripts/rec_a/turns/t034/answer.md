I will quantify the late-head perturbation's action effects and compare them with the early-head control. A useful localization should change transition timing in the ambiguous grasp states, not merely shrink every gripper value toward zero.
```python
p13=json.loads((W/"probe_13/results.json").read_text())
pm13=[]; mlp13_orig=[]; mlp13_edit=[]
for ii in range(16):
    dd={p["arm"]:p for p in p13 if p["idx"]==ii}
    ao=np.load(dd["orig"]["action"]); ae=np.load(dd["edit"]["action"])
    ob=torch.load(dd["orig"]["input"],map_location="cpu",weights_only=False)
    width=float(np.abs(ob["element"]["observation/state"][-2:]).sum())
    flips=int(np.sum((ao[:,6]>0)!=(ae[:,6]>0)))
    xrms=float(np.sqrt(np.mean((ao[:,:3]-ae[:,:3])**2)))
    grms=float(np.sqrt(np.mean((ao[:,6]-ae[:,6])**2)))
    pm13.append((flips,xrms,grms))
    print(ii,"width",round(width,3),"sign changes",flips,"xyz/grip",round(xrms,4),round(grms,4),
          "".join("+" if v>0 else "-" for v in ao[:,6]),"->",
          "".join("+" if v>0 else "-" for v in ae[:,6]))
    mlp13_orig.append({k:np.asarray(v,dtype=np.float32) for k,v in np.load(dd["orig"]["activations"]).items() if k.startswith(("mean::","calls::"))})
    mlp13_edit.append({k:np.asarray(v,dtype=np.float32) for k,v in np.load(dd["edit"]["activations"]).items() if k.startswith(("mean::","calls::"))})
print("Total sign changes:",sum(x[0] for x in pm13),"/160")
print("Late MLP response (relative change) at each denoising call, averaged over observations:")
for l in [3,9,14,15,16,17]:
    k=f"calls::expert.{l}.mlp_out::action"
    xo=np.stack([a[k] for a in mlp13_orig]); xe=np.stack([a[k] for a in mlp13_edit])
    rr=np.linalg.norm(xe-xo,axis=2)/(np.linalg.norm(xo,axis=2)+1e-9)
    print(l,ascii_round(rr.mean(0),3))
```
