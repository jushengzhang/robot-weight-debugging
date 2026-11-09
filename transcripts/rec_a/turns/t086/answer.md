I will compare the empty-hand observations with their predicted vertical motion and review the rollout losses. If the policy already lifts an empty hand, adding more lift is the wrong correction; a sign-reversed causal test would then be more informative than simply increasing the dose.
```python
from collections import defaultdict
p28 = defaultdict(dict)
for r in r28: p28[(r["id"],r["grip"],r["seed"])][r["model"]]=r
print("Changed outcomes:")
for key,d in p28.items():
    a,b=d["original"],d["edited"]
    if a["success"]!=b["success"]:
        print(key, a["level"], a["instruction"], 
              "success",a["success"],b["success"],
              "flips",a["grip_flips"],b["grip_flips"],
              "path",a["eef_path_m"],b["eef_path_m"])
print("\nEmpty-like observations: id step width; XYZ; first5 Z/grip; predicted chunk grip")
for m in manifest:
    if m["group"]!="fully_closed": continue
    p=torch.load(m["input"], weights_only=False, map_location="cpu")
    s=np.asarray(p["element"]["observation/state"])
    a=np.asarray(p["action_chunk"])
    print(m["id"],m["step"],round(m["width"],4),
          np.round(s[:3],3).tolist(),
          np.round(a[:5,2].mean(),3),np.round(a[:5,6].mean(),3),
          np.sign(a[:,6]).astype(int).tolist())
print("\nOriginal loop state traces: XYZ and finger gap every 50 steps")
for sid,grip in [(367,"open"),(753,"open"),(1296,"open"),(628,"open")]:
    ep = next((r for r in read_roll(1) if r["id"]==sid and r["grip"]==grip),None)
    if not ep: continue
    print(sid,grip,ep["success"],ep["grip_flips"],ep["instruction"])
    for path in sorted(pathlib.Path(ep["frames_dir"]).glob("*_inputs.pt")):
        p=torch.load(path, weights_only=False, map_location="cpu")
        s=np.asarray(p["element"]["observation/state"])
        a=np.asarray(p["action_chunk"])
        print(p["step"],np.round(s[:3],3).tolist(),round(abs(s[-2]-s[-1]),3),
              "aXYZ",np.round(a[:5,:3].mean(0),3).tolist(),
              "grip",np.sign(a[:5,6]).astype(int).tolist())
```
