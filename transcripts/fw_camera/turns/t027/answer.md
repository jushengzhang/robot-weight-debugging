The initial proprioception is identical within every task group. Lighting controls also change wrist-image appearance, so I will use the mild-versus-severe camera pairs as the cleaner first-state comparison and keep lighting mainly as a behavioral control. I will prepare exact replay copies and quantify layer shifts relative to each layer's normal variation before probing with a shared diffusion-noise seed.```python
replay_dir=W/"replay"
replay_dir.mkdir(exist_ok=True)
replay_paths=[]
for i,src in enumerate(diag_input_paths):
    dst=replay_dir/f"{i:02d}.pt"
    shutil.copy2(src,dst)
    replay_paths.append(str(dst))
camera_pair_diff=[]
for gi,dat in enumerate(init_data):
    dif=dat[1]["input_image"].float()-dat[0]["input_image"].float()
    camera_pair_diff.append([float(dif[:,:,:,:224].square().mean().sqrt()),
                             float(dif[:,:,:,224:].square().mean().sqrt())])
print("Mild/severe camera image RMS agent/wrist:")
print("mean:",np.round(np.mean(camera_pair_diff,axis=0),4).tolist(),
      "wrist maximum:",round(float(np.max(np.array(camera_pair_diff)[:,1])),4))
dA_cam=A_roll[:,1]-A_roll[:,0]
normal_task_sd=np.sqrt(np.mean((A_roll[:,2]-A_roll[:,2].mean(0))**2,axis=(0,2)))
shift_rms=np.sqrt(np.mean(dA_cam**2,axis=(0,2)))
print("Largest rollout camera-pair shifts / lighting between-task variation:")
for l in np.argsort(shift_rms/(normal_task_sd+1e-8))[-10:][::-1]:
    print(int(l),"shift",round(float(shift_rms[l]),4),"reference SD",round(float(normal_task_sd[l]),4),
          "ratio",round(float(shift_rms[l]/normal_task_sd[l]),3))
action_stats=[]
for r in roll1:
    a=np.load(r["actions_npy"])
    action_stats.append({"category":r["category"],"success":r["success"],
                         "xyz_rms":float(np.sqrt(np.mean(a[:,:3]**2))),
                         "rotation_rms":float(np.sqrt(np.mean(a[:,3:6]**2))),
                         "saturated_xyz":float((np.abs(a[:,:3])>.95).mean()),
                         "gripper_switches":int(np.sum(np.sign(a[1:,6])!=np.sign(a[:-1,6]))),
                         "steps":r["steps"]})
print("Rollout action summaries:")
for cat,success in [("Camera Viewpoints",False),("Camera Viewpoints",True),("Light Conditions",True)]:
    rows=[x for x in action_stats if x["category"]==cat and x["success"]==success]
    print(cat,success,"n",len(rows),
          {k:round(float(np.mean([x[k] for x in rows])),3)
           for k in ("xyz_rms","rotation_rms","saturated_xyz","gripper_switches","steps")})
(W/"roll1_action_statistics.json").write_text(json.dumps(action_stats,indent=2))
print("Probe inputs:",json.dumps(replay_paths,separators=(",",":")))
```
