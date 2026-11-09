The stronger MLP dose also failed to improve recovery, so I will revert it. The remaining distinction worth testing is not simply open versus closed, but fully closed versus object-separated fingers. I will collect stable examples of those visual states, plus open-hand controls, to test whether a failed-grasp cue could support corrective arm motion without weakening normal grasps.
```python
for n in list(edited()):
    drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 25: E16 MLP grip32 gain 1.5 rejected\n"
            "- Screen 31/64 -> 29/64, -3.1 pp (SE 3.1), 1 rescue / 3 losses; loops 21 -> 20; flips 12.39 -> 13.36.\n"
            "- Neither nonlinear-feedback dose improved recovery. Reverted.\n"
            "- Next hypothesis: fully closed/empty-like hand cue is present visually but insufficiently coupled to corrective arm motion. Distinguish it from object-separated closed fingers and open hand, rather than weakening grasp commands.\n")
physical_candidates={"fully_closed":[],"holding_like":[],"open":[]}
for r in r1:
    if r["success_at_t0"]:
        continue
    acts=np.load(local_path(r["actions_npy"]))
    for fp in sorted(local_path(r["frames_dir"]).glob("*_inputs.pt")):
        v=torch.load(fp,map_location="cpu",weights_only=False)
        state=np.asarray(v["element"]["observation/state"])
        width=float(state[6]+state[7])
        ac=np.asarray(v["action_chunk"])
        step=int(v["step"])
        prior=acts[max(0,step-10):step,6] if step>0 else np.array([])
        stable_close=(step>=50 and len(prior)>0 and np.mean(prior>0)>=.9 and np.mean(ac[:5,6]>0)>=.8)
        stable_open=(len(prior)==0 or np.mean(prior<=0)>=.8) and np.mean(ac[:5,6]<=0)>=.8
        group=None
        if stable_close and width<.008 and not r["success"] and r["grip_flips"]>=6:
            group="fully_closed"
        elif stable_close and .018<width<.067 and r["success"]:
            group="holding_like"
        elif stable_open and width>.074:
            group="open"
        if group:
            physical_candidates[group].append({"input":str(fp),"id":r["id"],"step":step,"width":width,
                                               "group":group,"suite":r["suite"],"grip":r["grip"],
                                               "instruction":r["instruction"],"success":r["success"]})
physical_panel=[]
for group,pool in physical_candidates.items():
    # Stable, deterministic scene diversification; no weight or reward optimization.
    pool.sort(key=lambda x:(x["step"],x["id"],x["grip"]))
    counts={}
    selected=[]
    for x in pool:
        if counts.get(x["id"],0)>=1:
            continue
        selected.append(x); counts[x["id"]]=1
        if len(selected)==16:
            break
    if len(selected)<16:
        for x in pool:
            if x not in selected and counts.get(x["id"],0)<2:
                selected.append(x); counts[x["id"]]=counts.get(x["id"],0)+1
                if len(selected)==16:
                    break
    physical_panel.extend(selected)
    print(group,"pool",len(pool),"selected",len(selected),"scene count",len(set(x["id"] for x in selected)),
          "width range", [round(min(x["width"] for x in selected),4),round(max(x["width"] for x in selected),4)] if selected else [])
(W/"physical_grip_probe_manifest.json").write_text(json.dumps(physical_panel,indent=2))
print("Input paths:")
print(json.dumps([x["input"] for x in physical_panel]))
notes=(W/"notes.md").read_text().replace("Forward passes used: 14/50; code runs before restart: 26/300.",
                                        "Forward passes used: 25/50. Current edits: none. All tested candidates rejected so far.")
(W/"notes.md").write_text(notes)
```
