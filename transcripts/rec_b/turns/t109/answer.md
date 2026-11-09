Amplifying mid-layer visual variation also hurt recovery, so I will reject that edit. One sharper test remains: separate useful scene changes from changes caused by the robot’s own fingers. The benchmark’s two gripper starts may provide near-counterfactual observations of the same scene. I will check that their initial arm poses match before using them to isolate a visual gripper-feedback direction.
```python
for n in list(edited()):
    drop_edit(n)

scene_by_id={s["id"]:s for s in scenes}
def original_input_path(sid,grip,step=0):
    scene_id=scene_by_id[sid]["scene_id"]
    return W/"fwd"/"roll_1"/"artifacts"/f"{scene_id}_{grip}_s17_orig_frames"/f"t{step:04d}_inputs.pt"

initial_pairs=[]
for sid in sorted(set(r["id"] for r in base_rows)):
    ps=[original_input_path(sid,g) for g in ["open","close"]]
    if not all(p.exists() for p in ps):
        continue
    ins=[torch.load(p,map_location="cpu",weights_only=False) for p in ps]
    states=[np.asarray(i["element"]["observation/state"]) for i in ins]
    sd=states[0]-states[1]
    imd={}
    for cam in ["image","wrist_image"]:
        ims=[np.asarray(i["element"]["observation/"+cam]).astype(float) for i in ins]
        imd[cam]=float(np.mean(np.abs(ims[0]-ims[1])))
    initial_pairs.append({"id":sid,"pose_delta":float(np.max(np.abs(sd[:6]))),
                          "finger_delta":float(np.max(np.abs(sd[6:]))),**imd})
print("Initial pairs:",len(initial_pairs))
print("Arm-pose difference min/median/max:",
      [round(float(v),6) for v in np.quantile([r["pose_delta"] for r in initial_pairs],[0,.5,1])])
print("Finger difference min/median/max:",
      [round(float(v),6) for v in np.quantile([r["finger_delta"] for r in initial_pairs],[0,.5,1])])
print("Examples:",initial_pairs[:8])
(W/"initial_gripper_pair_audit.json").write_text(json.dumps(initial_pairs))

with (W/"notes.md").open("a") as f:
    f.write("\n## N alpha2 rejected, passes42-43\n"
            "Midlayer intervention produces~.012 translation changes in two loops vs.0037/.0009 controls, "
            "but diagnostic57/96->52/96,-5.2+/-3.1pp,2 rescues/7 regressions,loops35->38. "
            "Amplifying generic visual variation is not a repair. Reset. "
            "Final focused diagnostic: isolate initial open/closed-finger image differences "
            "while arm pose and task are held fixed, to distinguish robot self-image from scene evidence.\n")
```
