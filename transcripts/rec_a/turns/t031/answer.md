The early-head ablation changes gripper amplitude more than the earlier interventions, but almost never changes its sign or transition timing. Before treating that as meaningful, I will check whether the change is merely a constant offset and how strongly the unedited gripper decisions depend on the saved flow noise. This guards against mistaking a readout bias for recovery behavior.
```python
print("Initial noise gripper coordinates:")
for g in ["open","close"]:
    n=np.asarray(initial_obs[(367,g)]["noise"])
    print(g,"shape",n.shape,"mean/std",round(float(n.mean()),3),round(float(n.std()),3),
          "gripper",ascii_round(n[:,6],3))
print("E0H6 half-output gripper delta mean/std:")
for ii in [0,1,2,3,4,5,6,7,12,13]:
    dd={p["arm"]:p for p in p12 if p["idx"]==ii}
    d=np.load(dd["edit"]["action"])[:,6]-np.load(dd["orig"]["action"])[:,6]
    print(ii,round(float(d.mean()),4),round(float(d.std()),4))
# Observational dependence of final actions on Gaussian gripper coordinate.
# Read only generated policy-input records, not any additional benchmark files.
all_inputs_meta=[]
for r in usable:
    for fp in sorted(Path(r["frames_dir"]).glob("*_inputs.pt")):
        ob=torch.load(fp,map_location="cpu",weights_only=False)
        ac=np.asarray(ob["action_chunk"]); no=np.asarray(ob["noise"])
        st=np.asarray(ob["element"]["observation/state"])
        all_inputs_meta.append({"id":r["id"],"grip":r["grip"],"step":ob["step"],
                                "success":r["success"],"loop":not r["success"] and r["grip_flips"]>=6,
                                "width":float(np.abs(st[-2:]).sum()),"noise_g":no[:,6].copy(),
                                "action_g":ac[:,6].copy(),"path":str(fp)})
for label,rr in [
    ("success",[z for z in all_inputs_meta if z["success"]]),
    ("loop",[z for z in all_inputs_meta if z["loop"]]),
    ("other",[z for z in all_inputs_meta if not z["success"] and not z["loop"]])]:
    ng=np.concatenate([z["noise_g"] for z in rr]); ag=np.concatenate([z["action_g"] for z in rr])
    nw=np.array([z["width"] for z in rr]); am=np.array([z["action_g"][:5].mean() for z in rr])
    print(label,"nobs",len(rr),"noise/output correlation",round(float(np.corrcoef(ng,ag)[0,1]),3),
          "width/output correlation",round(float(np.corrcoef(nw,am)[0,1]),3),
          "within-chunk flips",round(float(np.mean([np.sum((z["action_g"][1:]>0)!=(z["action_g"][:-1]>0)) for z in rr])),3))
# Current head ablation is diagnostic only; revert before the next localization.
clear_edits()
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 12: L0 H6 half-output diagnostic\n"
            "- 16 identical-input paired probes: 1/160 gripper signs changed; "
            "mean XYZ RMS 0.0072. Most gripper amplitude shifts ~0.03, not a timing repair.\n"
            "- No rollout claim made; reverted to investigate actual gripper-decision route.\n"
            "- Initial noise has only two arrays across discovery scenes, keyed by gripper start; "
            "this explains much of the very consistent expert initial contrast.\n")
```
