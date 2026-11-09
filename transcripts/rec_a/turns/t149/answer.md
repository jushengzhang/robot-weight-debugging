I will extract the cue actually read by the head from the first-call attention-output difference, then compare gripper and pose changes. This is a stronger test than the pooled-feature separation alone.
```python
r43=json.loads((W/"probe_43"/"results.json").read_text())
s43=[]
for i in range(16):
    pair=[next(r for r in r43 if r["idx"]==i and r["arm"]==a) for a in ["orig","edit"]]
    a,b=[np.load(r["action"]) for r in pair]
    az,bz=[np.load(r["activations"])["calls::expert.17.attn_out::action"] for r in pair]
    cue=float((torch.from_numpy(bz[0]-az[0]).float()@wg)/2)
    s43.append({"i":i,"type":("empty" if i<4 else "holding" if i<8 else tr[trinds[i-8]]["group"]+"_"+tr[trinds[i-8]]["direction"]),
                "attended_cue":round(cue,3),
                "grip_delta":round(float((b[:,6]-a[:,6]).mean()),4),
                "first5_grip_delta":round(float((b[:5,6]-a[:5,6]).mean()),4),
                "signs":int(((a[:,6]>0)!=(b[:,6]>0)).sum()),
                "xyz_rms":round(float(np.sqrt(np.mean((b[:,:3]-a[:,:3])**2))),5)})
print(json.dumps(s43,indent=1))
print("Signs:",sum(s["signs"] for s in s43),"/160","XYZ RMS avg:",round(np.mean([s["xyz_rms"] for s in s43]),5))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 43: late empty-hand opening feedback probe\n")
    for s in s43: f.write("- "+json.dumps(s)+"\n")
```
