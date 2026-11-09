L15 H4 also failed the rollout screen: one rescue versus three losses. I will revert it. The existing probe panel contains mostly saturated gripper commands, so I will build a panel of actual predicted transitions and check whether looping episodes close the gripper while substantial approach motion is still being commanded. That would motivate a targeted temporal-attention test.
```python
for n in list(edited()):
    drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 22: E15H4 half-output rejected\n"
            "- Screen 31/64 -> 29/64, -3.1 pp (SE 3.1), 1 rescue / 3 losses; loops 21 -> 19 but mean flips increased 12.55 -> 13.69.\n"
            "- Reverted. The previous probe panel was mostly saturated gripping states; next panel emphasizes predicted within-chunk transitions.\n")

# Analyze real executed traces; no simulator or model execution here.
transition_stats={"loop":[],"success":[]}
transition_candidates=[]
for r in r1:
    group="success" if r["success"] and not r["success_at_t0"] else ("loop" if not r["success"] and r["grip_flips"]>=6 else None)
    if group is None:
        continue
    act=np.load(local_path(r["actions_npy"]))
    if act.ndim==2 and act.shape[1]>=7:
        closing=np.flatnonzero((act[:-1,6]<=0)&(act[1:,6]>0))+1
        for t in closing:
            if group=="loop" and t<100:
                continue
            pre=act[max(0,t-3):t,:3]
            post=act[t:min(len(act),t+5),:3]
            transition_stats[group].append([float(np.linalg.norm(pre,axis=1).mean()),
                                            float(pre[:,2].mean()),
                                            float(np.linalg.norm(post,axis=1).mean()),
                                            float(post[:,2].mean())])
    fd=local_path(r["frames_dir"])
    for fp in sorted(fd.glob("*_inputs.pt")):
        v=torch.load(fp,map_location="cpu",weights_only=False)
        ac=np.asarray(v["action_chunk"])
        s=ac[:,6]>0
        flips=np.flatnonzero(s[1:]!=s[:-1])+1
        if len(flips)==0:
            continue
        t=int(flips[0])
        transition_candidates.append({"input":str(fp),"id":r["id"],"grip":r["grip"],
                                      "step":int(v["step"]),"group":group,
                                      "direction":"close" if s[t] else "open",
                                      "boundary":t,"level":r["level"],"success":r["success"]})

for group,x in transition_stats.items():
    x=np.asarray(x)
    print(group,"closing transitions",len(x),
          "mean [pre XYZ norm, pre Z, post XYZ norm, post Z]",np.round(x.mean(0),4).tolist())

# Fixed diagnostic panel: up to 16 per outcome/transition stratum, at most two per scene per stratum.
transition_panel=[]
for group in ("loop","success"):
    for direction in ("close","open"):
        pool=[x for x in transition_candidates if x["group"]==group and x["direction"]==direction]
        pool.sort(key=lambda x:(abs(x["boundary"]-3),x["id"],x["step"],x["grip"]))
        counts={}
        chosen=[]
        for x in pool:
            if counts.get(x["id"],0)>=2:
                continue
            chosen.append(x); counts[x["id"]]=counts.get(x["id"],0)+1
            if len(chosen)==16:
                break
        transition_panel.extend(chosen)
        print("Panel",group,direction,len(chosen),"available",len(pool))
(W/"transition_probe_manifest.json").write_text(json.dumps(transition_panel,indent=2))
print("Transition panel inputs:",len(transition_panel))
print(json.dumps([x["input"] for x in transition_panel]))
```
