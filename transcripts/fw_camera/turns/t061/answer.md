Some light-control variants have no numeric difficulty label. I will handle those explicitly and inspect their names, rather than treating the missing label as a severity value.
```python
print("Light controls with missing difficulty:",
      [{k:v[k] for k in ["suite","id","name","difficulty"]} for v in light_dis if v["difficulty"] is None][:3])
extra_groups=[]
for gi,((suite,task),vs) in enumerate(sorted(cam_groups.items())):
    def tiehash(v):
        return hashlib.sha256(f"broader_diag:{suite}:{v['id']}".encode()).hexdigest()
    low_pool=[v for v in vs if v["difficulty"] is not None and v["difficulty"]<=2] or vs
    high_pool=[v for v in vs if v["difficulty"] is not None and v["difficulty"]>=4] or vs
    low=sorted(low_pool,key=lambda v:((suite,v["id"]) in old_cam_keys,v["difficulty"] or 0,tiehash(v)))[0]
    hp=[v for v in high_pool if v["id"]!=low["id"]] or [v for v in vs if v["id"]!=low["id"]]
    high=sorted(hp,key=lambda v:((suite,v["id"]) in old_cam_keys,-(v["difficulty"] or 0),tiehash(v)))[0]
    lp=[v for v in light_dis if v["suite"]==suite and v["name"].startswith(task+"_")]
    if not lp:
        continue
    # Use labeled mild light controls; no inference about missing labels.
    labeled=[v for v in lp if v["difficulty"] is not None]
    if not labeled:
        continue
    ctrl=sorted(labeled,key=lambda v:(v["difficulty"],tiehash(v)))[0]
    extra_groups.append({"suite":suite,"task":task,"seed":5100+gi,
                         "ids":[low["id"],high["id"],ctrl["id"]],
                         "difficulties":[low["difficulty"],high["difficulty"],ctrl["difficulty"]]})
extra_eps=[{"suite":g["suite"],"id":i,"seed":g["seed"]} for g in extra_groups for i in g["ids"]]
(W/"broad_diagnostic_groups.json").write_text(json.dumps(extra_groups,indent=2))
(W/"broad_diagnostic_episodes.json").write_text(json.dumps(extra_eps,indent=2))
print("Groups:",len(extra_groups),"episodes",len(extra_eps),
      "control difficulties",dict(Counter(g["difficulties"][2] for g in extra_groups)))
req11={"type":"rollout","model":"original","episodes":extra_eps,"save_frames":True}
(W/"request_11.json").write_text(json.dumps(req11))
print(json.dumps(req11,separators=(",",":")))
```
