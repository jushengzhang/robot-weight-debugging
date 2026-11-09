Halving the generated-gripper input mainly increased output magnitude (about 1.6 instead of 1.0) while leaving almost all reversal decisions unchanged, so I will remove it. A more relevant perceptual distinction may be fully closed, empty fingers versus fingers stopped around an object. I will inspect the recorded finger joint positions—used only for diagnosis, not supplied to the policy—to see whether the saved observations can test that distinction.
```python
drop_edit("action_in_proj.weight")
with (W/"notes.md").open("a") as f:
    f.write("\nPass31 rejects generated-gripper half feedback: grip magnitude rises to~1.6, "
            "but later-loop sign change only0.6%, flip counts unchanged; XYZ RMS~0.009. Removed without rollout.\n")
obsmeta=[]
for rk in [1,4]:
    for r in read_roll(rk):
        if r.get("model")!="original" or not r.get("frames_dir"): continue
        for p in sorted(Path(r["frames_dir"]).glob("*inputs.pt")):
            v=torch.load(p,map_location="cpu",weights_only=False)
            s=np.asarray(v["element"]["observation/state"],dtype=float)
            ac=np.asarray(v["action_chunk"])
            obsmeta.append({"id":r["id"],"grip":r["grip"],"seed":r["seed"],
                "input":str(p),"step":int(v["step"]), "success":r["success"],
                "loop":not r["success"] and r["grip_flips"]>=6,"suite":r["suite"],
                "state":s.tolist(),"width":float(np.abs(s[-2:]).sum()),
                "z":float(s[2]),"command_close":float(np.mean(ac[:5,6]>0))})
print("saved observations",len(obsmeta),"scenes",len(set(m["id"] for m in obsmeta)))
ss=np.array([m["state"] for m in obsmeta])
print("state dimensional quantiles")
for j in range(8):
    print(j,[round(float(x),4) for x in np.quantile(ss[:,j],[0,.1,.25,.5,.75,.9,1])])
for role in ["success","loop","other"]:
    ms=[m for m in obsmeta if (m["success"] if role=="success" else m["loop"] if role=="loop" else not m["success"] and not m["loop"])]
    print(role,len(ms),"width quartiles",[round(float(x),4) for x in np.quantile([m["width"] for m in ms],[.1,.25,.5,.75,.9])],
          "height quartiles",[round(float(x),3) for x in np.quantile([m["z"] for m in ms],[.1,.25,.5,.75,.9])])
print("example states",[(m["id"],m["step"],[round(x,4) for x in m["state"]],m["command_close"])
                        for m in obsmeta[:4]])
(W/"observation_geometry.json").write_text(json.dumps(obsmeta))
```
