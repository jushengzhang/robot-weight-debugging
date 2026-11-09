The original traces show both empty-hand lifting and repeated reopening while the fingers remain separated, so “empty grasp needs more lift” is too narrow. I will measure whether failed episodes also show unstable gripper forecasts across replanning boundaries, and inspect the available activation detail before choosing the next intervention.
```python
print("Probe26 activation shapes:")
z=np.load(W/"probe_26/probe_orig_0_acts.npz")
print({k:z[k].shape for k in z.files})
print("Probe20 activation shapes:")
z=np.load(W/"probe_20/probe_orig_0_acts.npz")
print({k:z[k].shape for k in z.files})
roll1=read_roll(1)
stats=defaultdict(list)
for r in roll1:
    actions=np.load(r["actions_npy"])
    cat="success" if r["success"] else ("loop" if r["grip_flips"]>=6 else "other_fail")
    if r["success_at_t0"]: continue
    for p in sorted(pathlib.Path(r["frames_dir"]).glob("*_inputs.pt")):
        d=torch.load(p, weights_only=False, map_location="cpu")
        t=int(d["step"]); chunk=np.asarray(d["action_chunk"])
        if t+10>len(actions): continue
        pred=chunk[5:10,6]>0
        act=actions[t+5:t+10,6]>0
        match0=np.mean((chunk[:5,6]>0)==(actions[t:t+5,6]>0))
        s=np.asarray(d["element"]["observation/state"])
        stats[cat].append([np.mean(pred!=act),
                           np.any(pred!=act),
                           np.mean(np.abs(chunk[5:10,:3]-actions[t+5:t+10,:3])),
                           np.sum(np.diff(chunk[:,6]>0)!=0),
                           match0,
                           abs(s[-2]-s[-1])])
for c,rows in stats.items():
    arr=np.array(rows)
    print(c,"n",len(rows),
          "future sign mismatch/token, any/chunk, XYZ MAE, within-chunk flips, first5 match, gap:",
          np.round(arr.mean(0),4).tolist())
print("Probe maps examples:")
for k in [2,3,5,8,10]:
    p=W/f"probe_{k}"/"probe_orig_0_acts.npz"
    if p.exists():
        zz=np.load(p)
        print(k,{key:zz[key].shape for key in zz.files if "attn" in key or "layout" in key})
```
