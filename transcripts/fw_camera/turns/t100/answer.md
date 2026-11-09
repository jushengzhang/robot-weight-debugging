I’ll compare the ablated heads’ attention maps with the original maps on the two exact-state camera pairs. I’ll also measure the executed action changes, since broader attention alone would not establish a useful repair.
```python
r26=json.loads((root/"probe_26/results.json").read_text())
orig_maps=np.stack([np.load(root/"probe_3"/f"probe_orig_{i}_attn_L29.npy") for i in range(4)])
edit_maps=np.stack([np.load(root/"probe_26"/f"probe_edit_{i}_attn_L29.npy") for i in range(4)])
agent_cols=(np.arange(98)%14)<7

def map_summary(maps):
    video_mass=maps.sum(-1)
    p=maps/np.maximum(video_mass[...,None],1e-20)
    entropy=-(p*np.log(np.maximum(p,1e-20))).sum(-1)/np.log(98)
    agent_fraction=p[...,agent_cols].sum(-1)
    return {"video_mass":video_mass.mean((0,2)),
            "entropy":entropy.mean((0,2)),
            "agent_fraction":agent_fraction.mean((0,2))}
mo,me=map_summary(orig_maps),map_summary(edit_maps)
print("Head, original->ablated [video_mass, agent_fraction, entropy]:")
for h in outlier_heads:
    print(h, np.round([mo[x][h] for x in ("video_mass","agent_fraction","entropy")],3).tolist(),
          np.round([me[x][h] for x in ("video_mass","agent_fraction","entropy")],3).tolist())
other=[h for h in range(24) if h not in outlier_heads]
print("Other-head mean absolute map change:",float(np.abs(edit_maps[:,other]-orig_maps[:,other]).mean()))
bacts=np.stack([np.load(r["action"]) for r in b3])
eacts=np.stack([np.load(r["action"]) for r in r26])
print("Per-axis first10 edit RMS:",np.round(np.sqrt(np.mean((eacts[:,:10]-bacts[:,:10])**2,axis=(0,1))),6).tolist())
pair_stat=[]
for i,j in [(0,1),(2,3)]:
    bo=float(np.mean((bacts[i,:10,:3]-bacts[j,:10,:3])**2)**.5)
    ce=float(np.mean((eacts[i,:10,:3]-eacts[j,:10,:3])**2)**.5)
    pair_stat.append({"original":bo,"ablated":ce,"relative_pct":100*(ce/bo-1)})
print("Camera-pair XYZ RMS:",pair_stat)
summary26={"suspect_original_entropy":float(mo["entropy"][outlier_heads].mean()),
           "suspect_ablated_entropy":float(me["entropy"][outlier_heads].mean()),
           "suspect_original_agent_fraction":float(mo["agent_fraction"][outlier_heads].mean()),
           "suspect_ablated_agent_fraction":float(me["agent_fraction"][outlier_heads].mean()),
           "pair_action":pair_stat}
(root/"probe26_analysis.json").write_text(json.dumps(summary26,indent=2))
(root/"request_27.json").write_text(json.dumps(req25))
print("Prepared pass27: same 128 discovery episodes as pass25, coordinate-ablation edit only.")
```
