I will quantify both the mean and the spread of the action changes. A near-zero control mean could hide large changes of opposite signs, so I will also check control-state vertical RMS, other arm coordinates, and gripper-sign changes.
```python
def probe_actions(k):
    rows = json.loads((W/f"probe_{k}"/"results.json").read_text())
    arms = {}
    for arm in sorted(set(r["arm"] for r in rows)):
        rr = sorted([r for r in rows if r["arm"] == arm], key=lambda r:r["idx"])
        arms[arm] = np.stack([np.load(r["action"]) for r in rr])
    return arms

a43 = probe_actions(43)
print("arms", list(a43), "shape", {k:list(v.shape) for k,v in a43.items()})
orig_key = next(k for k in a43 if k.startswith("orig"))
edit_key = next(k for k in a43 if not k.startswith("orig"))
d43 = a43[edit_key] - a43[orig_key]
group_results = {}
for group in dict.fromkeys(group_key(r) for r in m):
    inds = [i for i,r in enumerate(m) if group_key(r)==group]
    d = d43[inds, :5]
    dz_scene = d[:,:,2].mean(1)
    result = {
        "n":len(inds), "dz_mean":round(float(dz_scene.mean()),4),
        "dz_RMS":round(float(np.sqrt(np.mean(d[:,:,2]**2))),4),
        "dz_scene_p10_p90":[round(float(x),4) for x in np.quantile(dz_scene,[.1,.9])],
        "xy_RMS":round(float(np.sqrt(np.mean(d[:,:,:2]**2))),4),
        "rot_RMS":round(float(np.sqrt(np.mean(d[:,:,3:6]**2))),4),
        "grip_sign_change":round(float(np.mean((a43[orig_key][inds,:5,6]>0)!=(a43[edit_key][inds,:5,6]>0))),4)
    }
    group_results[group] = result
    print(group, result)
print("Other edit count:", len(edited()))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass43: attention-centered clearance0.60 fixed-input probe\n")
    for k,v in group_results.items():
        f.write(k+" "+json.dumps(v)+"\n")
    f.write("Installed edit is rank2 VLM17 v_proj only; relative Frobenius change2.774. "
            "Control means alone are not a safety guarantee; inspect RMS and held-out behavior.\n")
```
