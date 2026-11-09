I will compare the paired predictions for gripper-sign changes and pose changes. If the effect is selective rather than broadly disruptive, the next test will be a paired rollout on the fixed discovery panel.
```python
def probe_pair_arrays(k):
    recs = probe_records(k)
    rs = {(r["idx"],r["arm"]):r for r in recs}
    ids=sorted(set(r["idx"] for r in recs))
    orig=np.stack([np.load(local_path(rs[i,"orig"]["action"])) for i in ids])
    edit=np.stack([np.load(local_path(rs[i,"edit"]["action"])) for i in ids])
    return rs,ids,orig,edit

rs16,idx16,orig16,edit16 = probe_pair_arrays(16)
print("Input; group; original/edited close-command tokens; sign changes; XYZ RMS; rotation RMS")
for i in idx16:
    og=orig16[i,:,6]; eg=edit16[i,:,6]
    print(i, "empty" if i<8 or i==16 else "hold",
          int((og>0).sum()),int((eg>0).sum()),int(((og>0)!=(eg>0)).sum()),
          round(float(np.sqrt(np.mean((orig16[i,:,:3]-edit16[i,:,:3])**2))),4),
          round(float(np.sqrt(np.mean((orig16[i,:,3:6]-edit16[i,:,3:6])**2))),4))
print("Grip values range:",float(orig16[:,:,6].min()),float(orig16[:,:,6].max()),
      "non-endpoint fraction",float(np.mean(np.abs(np.abs(orig16[:,:,6])-1)>0.001)))
print("Aggregate: grip signs changed",int(((orig16[:,:,6]>0)!=(edit16[:,:,6]>0)).sum()),
      "/170; XYZ RMS",round(float(np.sqrt(np.mean((orig16[:,:,:3]-edit16[:,:,:3])**2))),4))
for group,ids in [("empty",list(range(8))+[16]),("hold",list(range(8,16)))]:
    print(group,"sign-change fraction",round(float(np.mean((orig16[ids,:,6]>0)!=(edit16[ids,:,6]>0))),3),
          "mean grip delta",round(float((edit16[ids,:,6]-orig16[ids,:,6]).mean()),3))

screen = json.loads((W/"screen_episodes.json").read_text())
print("Fixed screen episodes:",json.dumps(screen))
with (W/"notes.md").open("a") as f:
    f.write(f"\n## Pass 16: distributed visual-gripper attenuation probe\n"
            f"- 17 identical-input paired probes; {int(((orig16[:,:,6]>0)!=(edit16[:,:,6]>0)).sum())}/170 gripper signs changed.\n"
            f"- XYZ RMS change {np.sqrt(np.mean((orig16[:,:,:3]-edit16[:,:,:3])**2)):.4f}.\n")
```
