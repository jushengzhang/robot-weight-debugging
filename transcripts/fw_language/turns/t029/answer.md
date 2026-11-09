The rollout's frame directory is present, but the advertised input filename pattern did not match any files. I will inspect the actual artifact names and follow only paths inside WORK before constructing the probes.
```python
print("WORK top-level:",[p.name for p in W.iterdir()])
fp=Path(roll1[0]["frames_dir"])
print("Frames directory exists:",fp.exists(),"symlink:",fp.is_symlink())
print("Frame directory entries:",[p.name for p in fp.iterdir()][:20] if fp.exists() else [])
print("roll_1_frames entries:",[p.name for p in (W/"roll_1_frames").iterdir()][:5])
print("Fwd rollout entries:",[p.name for p in (W/"fwd"/"roll_1").iterdir()])
act0=np.load(roll1[0]["activations"])
print("Activation arrays:",[(k,act0[k].shape) for k in act0.files[:9]],"total",len(act0.files))
print("Actions shape:",np.load(roll1[0]["actions_npy"]).shape)
control_eps=[]
control_map={}
for r in roll1:
    task=r["name"].split("_language_")[0]
    choices=by_task_cat[(r["suite"],task,"Light Conditions")]
    c=min(choices,key=lambda v:(v["difficulty"],v["id"]))
    key=(c["suite"],c["id"],r["seed"])
    if key not in control_map:
        control_map[key]=c
        control_eps.append({"suite":c["suite"],"id":c["id"],"seed":r["seed"]})
print("Matched controls:",len(control_eps))
print(json.dumps(control_eps,separators=(",",":")))
(W/"diagnostic_control_episodes.json").write_text(json.dumps(control_eps))
```
