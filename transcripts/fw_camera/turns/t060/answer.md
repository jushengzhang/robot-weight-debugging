There are 39 discovery tasks with camera variants. I will build a broader matched panel using a mild and a severe view per task, plus the mildest available light control. The reserved validation variants will remain untouched.
```python
old_cam_keys = {(r["suite"],r["id"]) for r in [json.loads(s) for s in (W/"roll_1.jsonl").read_text().splitlines()] if r["category"]=="Camera Viewpoints"}
light_dis = [v for v in pv if v["category"]=="Light Conditions" and (v["suite"],int(v["id"])) in discovery_keys]
extra_groups=[]
for gi,((suite,task),vs) in enumerate(sorted(cam_groups.items())):
    low_pool=[v for v in vs if v["difficulty"]<=2] or vs
    high_pool=[v for v in vs if v["difficulty"]>=4] or vs
    def tiehash(v):
        return hashlib.sha256(f"broader_diag:{suite}:{v['id']}".encode()).hexdigest()
    low=sorted(low_pool,key=lambda v:((suite,v["id"]) in old_cam_keys,v["difficulty"],tiehash(v)))[0]
    hp=[v for v in high_pool if v["id"]!=low["id"]]
    if not hp:
        hp=[v for v in vs if v["id"]!=low["id"]]
    high=sorted(hp,key=lambda v:((suite,v["id"]) in old_cam_keys,-v["difficulty"],tiehash(v)))[0]
    lp=[v for v in light_dis if v["suite"]==suite and v["name"].startswith(task+"_")]
    if not lp:
        continue
    ctrl=sorted(lp,key=lambda v:(v["difficulty"],tiehash(v)))[0]
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
