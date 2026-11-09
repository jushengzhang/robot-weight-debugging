The probe records module outputs averaged over action tokens, not individual head outputs. I will use those outputs to locate sensitive layers, then use attention maps to identify the heads within them. I will not infer head activity from an underdetermined inverse of the output projection.

```python
pair_action_delta=np.array([
    np.sqrt(np.mean((probe3_actions[(p["suite"],p["high"])]
                   -probe3_actions[(p["suite"],p["low"])])**2)) for p in probe_pairs])
fixed_contrasts={}
fixed_stats=[]
for expert,cam in [("action","action"),("video","agent"),("video","wrist")]:
    for l in range(30):
        key=f"mean::{expert}.{l}.attn_out::{cam}"
        X=np.stack([probe3_acts[(p["suite"],p["high"])][key] for p in probe_pairs])
        C=np.stack([probe3_acts[(p["suite"],p["low"])][key] for p in probe_pairs])
        D=X-C
        uu,ss,vh=np.linalg.svd(D,full_matrices=False)
        dn=np.linalg.norm(D,axis=1)
        cn=np.linalg.norm(C,axis=1)
        task=np.linalg.norm(C-C.mean(0),axis=1)
        rho=np.corrcoef(dn,pair_action_delta)[0,1]
        fixed_stats.append({"expert":expert,"camera":cam,"layer":l,
                            "delta_rms":float(np.sqrt(np.mean(D**2))),
                            "relative_delta":float(np.median(dn/np.maximum(cn,1e-9))),
                            "shift_to_task":float(np.mean(dn**2)/np.mean(task**2)),
                            "top1":float(ss[0]**2/(ss**2).sum()),
                            "top4":float((ss[:4]**2).sum()/(ss**2).sum()),
                            "action_delta_corr":float(rho)})
        fixed_contrasts[(expert,cam,l)]={"high":X,"low":C,"delta":D,
                                        "basis":vh[:8].T.copy(),"singular_values":ss}
print("Action fixed-input layer statistics:")
for x in fixed_stats:
    if x["expert"]=="action":
        print(x["layer"],"rms",round(x["delta_rms"],3),"relative",round(x["relative_delta"],3),
              "shift/task",round(x["shift_to_task"],3),"rank1",round(x["top1"],3),
              "rank4",round(x["top4"],3),"action_corr",round(x["action_delta_corr"],3))
print("Video fixed-input representative layers (camera,L,delta RMS,relative,top1):")
for x in fixed_stats:
    if x["expert"]=="video" and x["layer"] in [0,1,2,4,8,12,16,20,24,28,29]:
        print(x["camera"],x["layer"],round(x["delta_rms"],3),round(x["relative_delta"],3),round(x["top1"],3))
print("Pair action changes: median",round(float(np.median(pair_action_delta)),4),
      "max",round(float(pair_action_delta.max()),4))
print("Largest action changes:",
      [(probe_pairs[i]["suite"],probe_pairs[i]["high"],probe_pairs[i]["low"],round(float(pair_action_delta[i]),4))
       for i in np.argsort(-pair_action_delta)[:8]])
(W/"fixed_input_layer_stats.json").write_text(json.dumps(fixed_stats,indent=2))
np.savez(W/"fixed_input_contrasts.npz",
         **{f"{e}_{c}_{l}_{k}":v for (e,c,l),ss in fixed_contrasts.items() for k,v in ss.items()})
```
