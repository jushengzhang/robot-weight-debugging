I will compare each camera episode with its same-task lighting control at both the rollout and first-observation level. In particular, I will check whether the wrist image and proprioception remain stable while the agent image changes, and identify where attention-output differences grow through the network. This avoids attributing ordinary state differences to camera sensitivity.```python
rec_by_key={(r["suite"],r["id"],r["seed"]):r for r in roll1}
r0=roll1[0]
with np.load(r0["activations"]) as z:
    print("Activation keys:",z.files[:10])
    print("First shapes:",[(k,z[k].shape) for k in z.files[:6]])
a0=np.load(r0["actions_npy"])
print("Rollout action-array shape:",a0.shape,"range:",float(a0.min()),float(a0.max()))
del a0

def act_key(ex,l,cam="action"):
    return f"mean::{ex}.{l}.attn_out::{cam}"

A_roll=[]; V_roll=[]
for x in diag_variants:
    r=rec_by_key[(x["suite"],x["id"],x["seed"])]
    with np.load(r["activations"]) as z:
        A_roll.append(np.stack([z[act_key("action",l)] for l in range(30)]))
        V_roll.append(np.stack([np.stack([z[act_key("video",l,c)] for c in ("agent","wrist")])
                                for l in range(30)]))
A_roll=np.stack(A_roll).reshape(-1,3,30,1024)
V_roll=np.stack(V_roll).reshape(-1,3,30,2,3072)
# Task-centered paired deviations; trajectory means are descriptive, not causal.
dA_lo=A_roll[:,0]-A_roll[:,2]
dA_hi=A_roll[:,1]-A_roll[:,2]
dV_lo=V_roll[:,0]-V_roll[:,2]
dV_hi=V_roll[:,1]-V_roll[:,2]
print("Layer: action RMS low/high; video high RMS agent/wrist")
for l in range(30):
    print("%02d %.4f %.4f %.4f %.4f" % (
        l,np.sqrt(np.mean(dA_lo[:,l]**2)),np.sqrt(np.mean(dA_hi[:,l]**2)),
        np.sqrt(np.mean(dV_hi[:,l,0]**2)),np.sqrt(np.mean(dV_hi[:,l,1]**2))))
init_data=[]
diag_input_paths=[]
for g in diag_groups:
    group=[]
    for idv in g["ids"]:
        r=rec_by_key[(g["suite"],idv,g["seed"])]
        path=Path(r["frames_dir"])/"t0000_inputs.pt"
        q=torch.load(path,map_location="cpu",weights_only=False)
        group.append(q)
        diag_input_paths.append(str(path))
    init_data.append(group)
print("Initial matched-state/image/action summaries:")
pair_initial_stats=[]
for gi,(g,dat) in enumerate(zip(diag_groups,init_data)):
    ref=dat[2]
    for severity,j in (("low",0),("high",1)):
        r=rec_by_key[(g["suite"],g["ids"][j],g["seed"])]
        imgdiff=dat[j]["input_image"].float()-ref["input_image"].float()
        actiondiff=np.asarray(dat[j]["action_chunk"])-np.asarray(ref["action_chunk"])
        row={"group":gi,"suite":g["suite"],"id":g["ids"][j],"severity":severity,
             "success":bool(r["success"]),
             "proprio_maxdiff":float((dat[j]["proprio"]-ref["proprio"]).abs().max()),
             "agent_image_rms":float(imgdiff[:,:,:,:224].square().mean().sqrt()),
             "wrist_image_rms":float(imgdiff[:,:,:,224:].square().mean().sqrt()),
             "first10_xyz_action_rms":float(np.sqrt(np.mean(actiondiff[:10,:3]**2)))}
        pair_initial_stats.append(row)
    outcomes=[rec_by_key[(g["suite"],idv,g["seed"])]["success"] for idv in g["ids"]]
    print(gi,g["suite"],g["seed"],"low/high/light successes:",outcomes,
          "high camera first10 xyz RMS:",round(pair_initial_stats[-1]["first10_xyz_action_rms"],3))
(W/"initial_pair_statistics.json").write_text(json.dumps(pair_initial_stats,indent=2))
print("Matched initial proprio maxdiff:",max(x["proprio_maxdiff"] for x in pair_initial_stats))
print("Mean camera image RMS agent/wrist:",
      np.round([np.mean([x[k] for x in pair_initial_stats])
                for k in ("agent_image_rms","wrist_image_rms")],4).tolist())
print("Probe input list saved, count:",len(diag_input_paths))
(W/"diagnostic_input_paths.json").write_text(json.dumps(diag_input_paths,indent=2))
```
