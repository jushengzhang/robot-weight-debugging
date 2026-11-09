I will compare gripper signs, amplitudes, and pose changes, and inspect the recorded residual differences. This will show whether the weight-based closing signature predicts the actual policy response.
```python
r41=json.loads((W/"probe_41"/"results.json").read_text())
s41=[]
for i in range(16):
    pair=[next(r for r in r41 if r["idx"]==i and r["arm"]==a) for a in ["orig","edit"]]
    a,b=[np.load(r["action"]) for r in pair]
    s41.append({"i":i,"type":("empty" if i<4 else "holding" if i<8 else tr[trinds[i-8]]["group"]+"_"+tr[trinds[i-8]]["direction"]),
                "grip_delta":round(float((b[:,6]-a[:,6]).mean()),4),
                "grip_first5_delta":round(float((b[:5,6]-a[:5,6]).mean()),4),
                "signs":int(((a[:,6]>0)!=(b[:,6]>0)).sum()),
                "xyz_rms":round(float(np.sqrt(np.mean((b[:,:3]-a[:,:3])**2))),5)})
print(json.dumps(s41,indent=1))
print("All sign changes:",sum(s["signs"] for s in s41),"/160",
      "mean XYZ RMS:",round(float(np.mean([s["xyz_rms"] for s in s41])),5))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 41: E15 empty-closing8 gain 0.5 probe\n")
    for s in s41: f.write("- "+json.dumps(s)+"\n")
```
