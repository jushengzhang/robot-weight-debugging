I will quantify the attention and action changes before deciding whether this conditional edit deserves a rollout. In particular, I will check whether empty-hand inputs gain visual attention without comparably large changes in the holding controls.
```python
r40=json.loads((W/"probe_40"/"results.json").read_text())
s40=[]
for i in range(4):
    pair=[next(r for r in r40 if r["idx"]==i and r["arm"]==a) for a in ["orig","edit"]]
    att=[]
    for r in pair:
        x=np.load(r["attention"]["16"])[3].mean(0)
        att.append([round(float(x[:256].sum()),4),round(float(x[256:512].sum()),4),
                    round(float(x[768:].sum()),4),round(float(1-x.sum()),4)])
    a,b=[np.load(r["action"]) for r in pair]
    s40.append({"input":i,"group":phys[selected[i]]["group"],"base_wrist_lang_self":att,
                "xyz_rms":round(float(np.sqrt(np.mean((b[:,:3]-a[:,:3])**2))),5),
                "first5_xyz_delta":np.round((b[:5,:3]-a[:5,:3]).mean(0),4).tolist(),
                "grip_sign_changes":int(((a[:,6]>0)!=(b[:,6]>0)).sum())})
print(json.dumps(s40,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 40: conditional empty-hand query probe\n")
    for s in s40: f.write("- "+json.dumps(s)+"\n")
```
