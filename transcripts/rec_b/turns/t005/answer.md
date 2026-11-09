I will quantify the difference between repetitive motion and useful progress, then rank layers by loop-associated changes without treating those correlations as causal. I will also select matched success/failure observations for attention-map probes, especially scenes where changing only the initial gripper state changes the outcome.
```python
R={ (r["id"],r["grip"]):r for r in r1}
A={k:dict(np.load(r["activations"])) for k,r in R.items() if r.get("activations")}
print("activation keys/shapes:",[(k,v.shape) for k,v in next(iter(A.values())).items()][:14])
valid=[r for r in r1 if not r["success_at_t0"]]
for label,rr in [("success",[r for r in valid if r["success"]]),
                 ("loop",[r for r in valid if not r["success"] and r["grip_flips"]>=6]),
                 ("other_fail",[r for r in valid if not r["success"] and r["grip_flips"]<6])]:
    print(label,len(rr),{k:round(float(np.median([r[k] for r in rr])),3)
          for k in ["grip_flips","steps","eef_path_m","eef_net_m","frac_grip_closed"]},
          "median net/path",round(float(np.median([r["eef_net_m"]/max(r["eef_path_m"],1e-6) for r in rr])),3))
print("discordant grips:",[(s, R[s,"open"]["success"],R[s,"close"]["success"],by_id[s]["level"])
                           for s in diag_ids if R[s,"open"]["success"] != R[s,"close"]["success"]])
print("L3 records:",[(r["id"],r["grip"],r["success"],r["grip_flips"],r["steps"])
                     for r in r1 if r["level"]=="L3"])
akeys=sorted(next(iter(A.values())).keys())
out_w=load_original("action_out_proj.weight")
print("action readout norms",out_w.norm(dim=1)[:10].numpy().round(3).tolist())
for key in akeys:
    if not ("expert" in key or "vlm" in key): continue
    rr=[r for r in valid if (r["id"],r["grip"]) in A and key in A[r["id"],r["grip"]]]
    X=np.stack([A[r["id"],r["grip"]][key].reshape(-1) for r in rr])
    ys=np.array([r["success"] for r in rr])
    yl=np.array([not r["success"] and r["grip_flips"]>=6 for r in rr])
    ms,ml=X[ys].mean(0),X[yl].mean(0)
    delta=ml-ms
    scale=np.sqrt(((X-X.mean(0))**2).sum(1).mean())
    metric=np.linalg.norm(delta)/max(scale,1e-8)
    if "expert" in key:
        gp=float(delta@out_w[6].numpy()/max(np.linalg.norm(delta)*float(out_w[6].norm()),1e-8))
        print(key,"effect_norm",round(float(metric),3),"grip_align",round(gp,3),
              "rms S/L",round(float(np.sqrt((X[ys]**2).mean())),3),round(float(np.sqrt((X[yl]**2).mean())),3))
print("example input/frame files:",[p.name for p in pathlib.Path(R[1102,"open"]["frames_dir"]).iterdir()][:10])
print("action shape",np.load(R[1102,"open"]["actions_npy"]).shape)
with (W/"notes.md").open("a") as f:
    f.write("\n## Baseline, pass 1\n96 episodes, 57 successes (59.4%), 7 already solved at t0. "
            "L1 16/24, L2 18/24, L3 8/24, L4 15/24. "
            "39 failures, 35 re-grasp loops. Original observations/actions and attention-output summaries saved.\n")
```
