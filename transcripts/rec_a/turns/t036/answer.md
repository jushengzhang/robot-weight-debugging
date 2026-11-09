I will broaden the identical-input test to recorded states where the fingers are physically closed with almost no gap, yet the predicted chunk keeps closing. I will include successful grasps with a nonzero finger gap as controls, so that strengthening feedback is not rewarded merely for opening every grasp.
```python
empty_loop=[z for z in all_inputs_meta if z["loop"] and z["width"]<0.008 and np.mean(z["action_g"][:5]>0)>=0.6]
held_success=[z for z in all_inputs_meta if z["success"] and 0.018<z["width"]<0.07 and np.mean(z["action_g"][:5]>0)>=0.6]
print("Available empty-closed looping snapshots:",len(empty_loop),"holding-like success controls:",len(held_success))
def unique_scene_snapshots(pool,n,seed=919):
    order=list(pool)
    rng=np.random.default_rng(seed)
    rng.shuffle(order)
    ans=[]; seen=set()
    for z in order:
        if z["id"] not in seen:
            ans.append(z); seen.add(z["id"])
        if len(ans)==n: break
    return ans
grip_probe_meta=unique_scene_snapshots(empty_loop,8)+unique_scene_snapshots(held_success,8,920)
# Ensure the previously localized empty-closed threshold observation is represented.
p13_meta=next(z for z in all_inputs_meta if z["id"]==1269 and z["grip"]=="close" and z["step"]==0)
if p13_meta["path"] not in [z["path"] for z in grip_probe_meta]:
    grip_probe_meta.append(p13_meta)
for z in grip_probe_meta:
    print(z["id"],z["grip"],z["step"],"loop" if z["loop"] else "success","width",round(z["width"],4),
          "g","".join("+" if v>0 else "-" for v in z["action_g"]))
grip_probe_paths=[str(Path(z["path"]).relative_to(W)) for z in grip_probe_meta]
(W/"gripper_probe_selection.json").write_text(json.dumps(grip_probe_paths))
print("Inputs:",json.dumps(["WORK/"+p for p in grip_probe_paths]))
```
