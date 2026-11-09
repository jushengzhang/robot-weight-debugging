Finger width alone can confuse a real grasp with a gripper that is still moving. I will therefore look for observations preceded by ten consecutive close commands, then pair fully closed and partially closed states within the same scene. This gives a more defensible test of contact-related visual cues without using success labels to define the contrast.
```python
record_map={(r["id"],r["grip"],r["seed"]):r for rk in [1,4] for r in read_roll(rk)
            if r.get("model")=="original"}
stable=[]
action_cache={}
for m in obsmeta:
    key=(m["id"],m["grip"],m["seed"]); t=m["step"]
    if t<10: continue
    if key not in action_cache: action_cache[key]=np.load(record_map[key]["actions_npy"])
    aa=action_cache[key]
    if len(aa)<t or not np.all(aa[t-10:t,6]>0): continue
    mm=dict(m)
    mm["contact_class"]="empty" if m["width"]<0.008 else "contact" if 0.018<m["width"]<0.068 else "other"
    stable.append(mm)
print("stable-close states",collections.Counter(m["contact_class"] for m in stable))
pairs=[]
for sid in sorted(set(m["id"] for m in stable)):
    es=[m for m in stable if m["id"]==sid and m["contact_class"]=="empty"]
    hs=[m for m in stable if m["id"]==sid and m["contact_class"]=="contact"]
    cand=[]
    for e in es:
        for h in hs:
            se,sh=np.array(e["state"]),np.array(h["state"])
            dist=np.linalg.norm(se[:3]-sh[:3])
            rot=np.linalg.norm(se[3:6]-sh[3:6])
            cand.append((dist+0.05*rot,dist,rot,e,h))
    if cand:
        b=min(cand,key=lambda v:v[0])
        pairs.append({"id":sid,"distance":b[1],"rotation_distance":b[2],"empty":b[3],"contact":b[4]})
print("paired scenes",len(pairs),"within10cm",sum(p["distance"]<.1 for p in pairs))
for p in sorted(pairs,key=lambda p:p["distance"])[:16]:
    print(p["id"],"dist",round(p["distance"],3),"rot",round(p["rotation_distance"],2),
          "steps",(p["empty"]["step"],p["contact"]["step"]),
          "widths",tuple(round(p[x]["width"],3) for x in ["empty","contact"]),
          "outcomes",(p["empty"]["success"],p["contact"]["success"]))
print("visual direction file keys",torch.load(W/"visual_grip_direction.pt",weights_only=False).keys())
(W/"contact_pairs.json").write_text(json.dumps(pairs))
```
