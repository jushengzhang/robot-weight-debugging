The traces include extreme cases with 96-121 gripper flips but only about 6.6 cm of net end-effector movement. I will quantify layer differences, select matched examples for attention probes, and reserve untouched scenes for later confirmation; pre-failure scenes will serve only as an ordinary-behavior proxy, not as a full original-LIBERO test.

```python
import hashlib
scene_by_id = {s["id"]: s for s in scenes}
valid1 = [r for r in r1 if r.get("termination") != "scene_error" and r.get("activations") and pathlib.Path(r["activations"]).exists()]
a0 = np.load(valid1[0]["activations"])
print("activation keys:", a0.files[:12], "count", len(a0.files))
print("first shapes:", [(k, a0[k].shape) for k in a0.files[:5]])
success1 = [r for r in valid1 if r["success"] and not r.get("success_at_t0")]
loop1 = [r for r in valid1 if not r["success"] and r["grip_flips"] >= 6]
other1 = [r for r in valid1 if not r["success"] and r["grip_flips"] < 6]
A1 = { (r["id"],r["grip"]): {k:np.asarray(np.load(r["activations"])[k], dtype=np.float32) for k in a0.files} for r in valid1}
def cosine(x,y):
    return float(np.dot(x.ravel(),y.ravel())/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12))
stats1=[]
for k in a0.files:
    x=np.stack([A1[(r["id"],r["grip"])][k] for r in success1])
    y=np.stack([A1[(r["id"],r["grip"])][k] for r in loop1])
    within=np.sqrt((x.var(0).sum()+y.var(0).sum())/2)
    stats1.append((k, float(np.linalg.norm(y.mean(0)-x.mean(0))/(within+1e-12)),
                   float(np.linalg.norm(x,axis=-1).mean()), float(np.linalg.norm(y,axis=-1).mean()),
                   cosine(x.mean(0),y.mean(0))))
print("largest standardized success-loop differences (key, distance, success norm, loop norm, cosine):")
for s in sorted(stats1,key=lambda t:-t[1])[:18]:
    print(s[0], *(round(v,4) for v in s[1:]))
groups1=collections.defaultdict(list)
for r in valid1:
    groups1[r["id"]].append(r)
matched1=[rs for rs in groups1.values() if len(rs)==2 and rs[0]["success"]!=rs[1]["success"]]
print("opposite-outcome same-scene pairs:")
for rs in matched1:
    print(rs[0]["id"],rs[0]["level"],rs[0]["instruction"],
          [(r["grip"],r["success"],r["grip_flips"],r["steps"]) for r in rs])
chosen_ids=[979,712,562,36]
print("candidate input paths:")
for id_ in chosen_ids:
    for r in groups1.get(id_,[]):
        fs=sorted(pathlib.Path(r["frames_dir"]).glob("*_inputs.pt"))
        print(id_,r["grip"],r["success"], [str(p.relative_to(W)) for p in fs[:2]+fs[-1:]])
p0=sorted(pathlib.Path(loop1[0]["frames_dir"]).glob("*_inputs.pt"))[0]
inp0=torch.load(p0,map_location="cpu",weights_only=False)
print("input type:", type(inp0).__name__)
if isinstance(inp0,dict):
    print("input fields:", {k:(list(v.shape) if hasattr(v,"shape") else type(v).__name__) for k,v in inp0.items()})
print("action shape:", np.load(loop1[0]["actions_npy"]).shape)
discovery_ids=set(r["id"] for r in r1)
remaining=[s for s in scenes if s["id"] not in discovery_ids]
def hsplit(s):
    # Split by source episode when its identifier follows the documented format.
    group=s["scene_id"].rsplit("__step",1)[0]
    return int(hashlib.sha256(("recovery-edit-v1:"+group).encode()).hexdigest()[:8],16)%100
dev_extra=[s for s in remaining if hsplit(s)<25]
validation=[s for s in remaining if 25<=hsplit(s)<60]
confirmation=[s for s in remaining if hsplit(s)>=60]
split_info={"discovery":sorted(discovery_ids),"development_extra":[s["id"] for s in dev_extra],
            "validation":[s["id"] for s in validation],"confirmation":[s["id"] for s in confirmation],
            "prefail_proxy":[s["id"] for s in scenes if s["stage"]=="pre-fail"]}
(W/"splits.json").write_text(json.dumps(split_info,indent=2))
print("split sizes:", {k:len(v) for k,v in split_info.items()})
```
