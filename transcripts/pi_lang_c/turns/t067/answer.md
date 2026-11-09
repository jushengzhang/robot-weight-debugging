For a controlled task-identity comparison, I will copy six saved goal observations and replace only their prompt with the corresponding literal instruction already recorded in the practice rollouts. Pixels and denoising noise will stay fixed. These diagnostic controls will be probed alongside the paraphrases and untouched object-layout inputs.
```python
ctrl_dir=W/"controlled_probes"
ctrl_dir.mkdir(exist_ok=True)
r1byid={r["id"]:r for r in read_roll(1)}
literal_set={r["instruction"] for r in read_roll(2)}
probe_cases=[]
for vid in [1420,1262,1179,1339,1299,1385]:
    r=r1byid[vid]
    src=W/"roll_1_frames"/f"libero_goal_v{vid}_s7_original"/"t0000_inputs.pt"
    literal=r["name"].split("_language_")[0].replace("_"," ")
    assert literal in literal_set
    data=torch.load(src,map_location="cpu",weights_only=False)
    control=copy.deepcopy(data)
    control["element"]["prompt"]=literal
    dst=ctrl_dir/f"goal_{vid}_literal.pt"
    torch.save(control,dst)
    assert np.array_equal(data["noise"],control["noise"])
    assert np.array_equal(data["element"]["observation/image"],control["element"]["observation/image"])
    probe_cases += [{"label":f"para_{vid}","path":str(src),"instruction":r["instruction"]},
                    {"label":f"literal_{vid}","path":str(dst),"instruction":literal}]
for vid in [1272,1191]:
    src=W/"roll_17_frames"/f"libero_object_v{vid}_s7_original"/"t0000_inputs.pt"
    data=torch.load(src,map_location="cpu",weights_only=False)
    probe_cases.append({"label":f"object_control_{vid}","path":str(src),"instruction":data["element"]["prompt"]})
r29=read_roll(29)
for suite in ["libero_goal","libero_object","libero_spatial","libero_10"]:
    seen=set()
    candidates=sorted([r for r in r29 if r["model"]=="original" and r["suite"]==suite],
                      key=lambda r: (-(r.get("difficulty") or 0),r["id"]))
    for r in candidates:
        if r["instruction"] in seen: continue
        seen.add(r["instruction"])
        src=pathlib.Path(r["frames_dir"])/"t0000_inputs.pt"
        assert src.exists()
        probe_cases.append({"label":f"layout_{suite}_{r['id']}","path":str(src),"instruction":r["instruction"]})
        if len(seen)==4: break
json.dump(probe_cases,(W/"F_controlled_probe_cases.json").open("w"),indent=2)
print("PROBE CASES",[(c["label"],c["instruction"]) for c in probe_cases[:14]])
print("PATHS",json.dumps([c["path"] for c in probe_cases],separators=(",",":")))
```
