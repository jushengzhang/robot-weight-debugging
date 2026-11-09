The two finger joints use opposite signs, so their sum is not the gripper aperture. I will correct the selection to use their separation before drawing any conclusion from these state labels.
```python
obs_catalog=[]
for r in r1:
    if r["success_at_t0"]:
        continue
    acts=np.load(local_path(r["actions_npy"]))
    for fp in sorted(local_path(r["frames_dir"]).glob("*_inputs.pt")):
        v=torch.load(fp,map_location="cpu",weights_only=False)
        st=np.asarray(v["element"]["observation/state"])
        ac=np.asarray(v["action_chunk"]); step=int(v["step"])
        prior=acts[max(0,step-10):step,6] if step>0 else np.array([])
        obs_catalog.append({"input":str(fp),"id":r["id"],"step":step,
                            "width":float(abs(st[6]-st[7])),
                            "q1":float(st[6]),"q2":float(st[7]),
                            "suite":r["suite"],"grip":r["grip"],"instruction":r["instruction"],
                            "success":r["success"],"flips":r["grip_flips"],
                            "command_close":float(np.mean(ac[:5,6]>0)),
                            "prior_close":float(np.mean(prior>0)) if len(prior) else None})
print("Finger joint / aperture quantiles:")
for k in ("q1","q2","width"):
    print(k,np.round(np.quantile([x[k] for x in obs_catalog],[0,.1,.5,.9,1]),4).tolist())

physical_candidates={"fully_closed":[],"holding_like":[],"open":[]}
for x in obs_catalog:
    stable_close=(x["step"]>=50 and x["prior_close"]>=.9 and x["command_close"]>=.8)
    stable_open=(x["prior_close"] is None or x["prior_close"]<=.2) and x["command_close"]<=.2
    group=None
    if stable_close and x["width"]<.008 and not x["success"] and x["flips"]>=6:
        group="fully_closed"
    elif stable_close and .018<x["width"]<.067 and x["success"]:
        group="holding_like"
    elif stable_open and x["width"]>.074:
        group="open"
    if group:
        physical_candidates[group].append(dict(x,group=group))

physical_panel=[]
for group,pool in physical_candidates.items():
    pool.sort(key=lambda x:(x["step"],x["id"],x["grip"]))
    counts={}; selected=[]
    for cap in (1,2):
        for x in pool:
            if x in selected or counts.get(x["id"],0)>=cap:
                continue
            selected.append(x); counts[x["id"]]=counts.get(x["id"],0)+1
            if len(selected)==16:
                break
        if len(selected)==16:
            break
    physical_panel.extend(selected)
    print(group,"pool",len(pool),"selected",len(selected),"scenes",len(set(x["id"] for x in selected)),
          "aperture", [round(min(x["width"] for x in selected),4),round(max(x["width"] for x in selected),4)] if selected else [])
(W/"physical_grip_probe_manifest.json").write_text(json.dumps(physical_panel,indent=2))
(W/"observation_catalog.json").write_text(json.dumps(obs_catalog,indent=2))
print(json.dumps([x["input"] for x in physical_panel]))
```
