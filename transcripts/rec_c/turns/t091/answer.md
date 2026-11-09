I will summarize the fixed-input effects before spending a rollout pass. A reduction in gripper reversals would support the proposed mechanism, but it would not be useful if it merely collapses gripper commands or substantially disrupts otherwise successful arm plans.
```python
p31=json.loads((W/"probe_31"/"results.json").read_text())
print("arms",collections.Counter(r["arm"] for r in p31))
by31={(r["idx"],r["arm"]):r for r in p31}
armnames=sorted(set(r["arm"] for r in p31))
ea=next(x for x in armnames if x!="orig")
m19=json.loads((W/"mlp_diagnostic_manifest.json").read_text())
stat31=collections.defaultdict(list)
for i,m in enumerate(m19):
    o=np.load(by31[i,"orig"]["action"])
    e=np.load(by31[i,ea]["action"])
    vals={"xyz_rms":np.sqrt(np.mean((o[:,:3]-e[:,:3])**2)),
          "grip_sign_change":np.mean((o[:,6]>0)!=(e[:,6]>0)),
          "orig_gabs":np.abs(o[:,6]).mean(),"edit_gabs":np.abs(e[:,6]).mean(),
          "orig_flips5":np.sum(np.diff(o[:5,6]>0)!=0),
          "edit_flips5":np.sum(np.diff(e[:5,6]>0)!=0),
          "orig_flips10":np.sum(np.diff(o[:,6]>0)!=0),
          "edit_flips10":np.sum(np.diff(e[:,6]>0)!=0)}
    stat31[m["phase"]+"_"+m["role"]].append(vals)
for label,rs in stat31.items():
    print(label,"n",len(rs),{k:round(float(np.mean([r[k] for r in rs])),4) for k in rs[0]})
print("original exact saved",sum(r.get("matches_saved_chunk",False) for r in p31 if r["arm"]=="orig"))
print("fwd files",[(q.name,q.is_dir()) for q in (W/"fwd"/"roll_1").iterdir()])
with (W/"notes.md").open("a") as f:
    f.write("\nPass31 fixed-input generated-gripper half feedback:\n")
    for label,rs in stat31.items():
        f.write(label+" "+str({k:round(float(np.mean([r[k] for r in rs])),4) for k in rs[0]})+"\n")
```
