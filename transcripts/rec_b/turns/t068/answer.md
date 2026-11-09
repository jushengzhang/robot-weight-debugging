I will compare the matched attention maps head by head, including the mass on action tokens rather than only the wrist camera. Any head selected for intervention must show a failure-related difference in these command-matched controls.
```python
m27=json.loads((W/"probe_27"/"results.json").read_text())
M27={}
for r in m27:
    print("Input",r["idx"],probe_identity(r),
          "mean action",np.round(np.load(localpath(r["action"])).mean(0),3).tolist())
    for l,p in r["attention"].items():
        a=np.load(localpath(p)).mean(1)
        M27[(r["idx"],int(l))]=np.stack([a[:,:256].sum(1),a[:,256:512].sum(1),
                                      a[:,768:].sum(1),1-a.sum(1)],axis=1)
print("idx0 F11 / idx1 S11 / idx2 F935 / idx3 S935")
for l in [7,12,15,16]:
    print("LAYER",l)
    for h in range(8):
        print(h,[[round(float(v),2) for v in M27[(i,l)][h]] for i in range(4)])
# More mechanical characterization of action traces: within-chunk vs between-chunk flips.
print("Flip sources in original diagnostic set")
def action_array(r):
    return np.load(localpath(r["actions_npy"]))
for lab,rs in [("loop",[r for r in rolls[1] if not r["success"] and r["grip_flips"]>=6]),
               ("success",[r for r in rolls[1] if r["success"] and not r.get("success_at_t0")])]:
    within=between=total=0;chunk_span=[]
    for r in rs:
        a=action_array(r)
        flips=np.flatnonzero(np.sign(a[1:,6])!=np.sign(a[:-1,6]))+1
        between+=int((flips%5==0).sum());within+=int((flips%5!=0).sum());total+=len(a)
        chunk_span.extend(np.diff(np.r_[0,flips,len(a)]).tolist())
    print(lab,"episodes",len(rs),"between_chunk",between,"within_chunk",within,
          "median_command_run",float(np.median(chunk_span)),"total_steps",total)
```
