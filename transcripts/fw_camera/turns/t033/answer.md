The probe activation file appears to expose action-layer outputs but not the per-camera video outputs available in rollout summaries. I will verify the available keys and restrict the first-state conclusions accordingly; camera-specific routing will come from the attention maps rather than an unavailable video activation measurement.```python
with np.load(p2raw[0]["activations"]) as z:
    all_probe_keys=list(z.files)
    print("Probe activation key count:",len(all_probe_keys))
    print("Non-action keys:",[k for k in all_probe_keys if "::action." not in k][:20])
P2A=[];P2act=[];P2norm=[]
for r in sorted(p2raw,key=lambda r:r["idx"]):
    with np.load(r["activations"]) as z:
        P2A.append(np.stack([z[act_key("action",l)] for l in range(30)]))
    P2act.append(np.load(r["action"]))
    P2norm.append(np.load(r["action_norm"]))
P2A=np.stack(P2A).reshape(19,3,30,1024)
P2act=np.stack(P2act).reshape(19,3,32,7)
P2norm=np.stack(P2norm).reshape(19,3,32,7)
dA=P2A[:,1]-P2A[:,0]
a_shift=np.sqrt(np.mean(dA**2,axis=(0,2)))
a_between=np.sqrt(np.mean((P2A[:,0]-P2A[:,0].mean(0))**2,axis=(0,2)))
print("First-state shifts: layer, action RMS, between-task SD, ratio")
for l in range(30):
    print("%02d %.4f %.4f %.3f" % (l,a_shift[l],a_between[l],a_shift[l]/a_between[l]))
pair_xyz_rms=np.sqrt(np.mean((P2act[:,1,:10,:3]-P2act[:,0,:10,:3])**2,axis=(1,2)))
print("Mean first10 camera-pair XYZ discrepancy:",round(float(pair_xyz_rms.mean()),4))
print("Largest first-state action discrepancies:")
for gi in np.argsort(pair_xyz_rms)[-8:][::-1]:
    g=diag_groups[gi]
    outs=[rec_by_key[(g["suite"],i,g["seed"])]["success"] for i in g["ids"]]
    print(int(gi),g["suite"],g["ids"],outs,"RMS",round(float(pair_xyz_rms[gi]),4))
print("Camera contrast rank concentration in selected action outputs:")
for l in (5,11,16,20,26,28,29):
    s=np.linalg.svd(dA[:,l],compute_uv=False)
    print(l,"top1/top4 energy",np.round([s[:k].dot(s[:k])/s.dot(s) for k in (1,4)],3).tolist())
np.savez(W/"firststate_pair_summary.npz",action_shift=a_shift,action_between=a_between,
         xyz_pair_rms=pair_xyz_rms)
with open(W/"notes.md","a") as f:
    f.write("\n## Pass 2: same-state probe\n"
            "- 19 mild/severe camera pairs have exactly identical wrist pixels and proprioception.\n"
            "- Original first10 XYZ camera-pair RMS discrepancy (mean across pairs): "
            +f"{pair_xyz_rms.mean():.4f}.\n"
            "- Probe activations expose action outputs only; per-camera video outputs are rollout means.\n"
            "- Failed camera rollouts have mean 17.56 gripper switches vs 5.80 successful camera "
            "and 2.00 successful light episodes; all failures timed out (not saturated actions).\n")
```
