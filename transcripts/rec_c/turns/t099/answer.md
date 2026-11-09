I will quantify the probe effects by empty closure versus contact width and by source versus reserved scenes. Then I will use paired rollouts to determine whether those local changes actually improve recovery; feature separation alone is not evidence of a repair.
```python
p33=json.loads((W/"probe_33"/"results.json").read_text())
b33={(r["idx"],r["arm"]):r for r in p33}
sg=collections.defaultdict(list)
for m in contact_manifest:
    i=m["index"]
    o=np.load(b33[i,"orig"]["action"]); e=np.load(b33[i,"edit"]["action"])
    sg[m["split"]+"_"+m["class"]].append({
        "xyz_rms":np.sqrt(np.mean((e[:5,:3]-o[:5,:3])**2)),
        "delta_z":(e[:5,2]-o[:5,2]).mean(),
        "orig_close":(o[:5,6]>0).mean(),"edit_close":(e[:5,6]>0).mean(),
        "sign_changes":((o[:5,6]>0)!=(e[:5,6]>0)).mean(),
        "delta_arm_norm":np.linalg.norm(e[:5,:3],axis=1).mean()-np.linalg.norm(o[:5,:3],axis=1).mean()})
for lab,rs in sg.items():
    print(lab,len(rs),{k:round(float(np.mean([r[k] for r in rs])),4) for k in rs[0]})
practice_data=json.loads((Path(BENCH)/"practice_scenes.json").read_text())
print("practice structure",type(practice_data).__name__,
      list(practice_data)[:5] if isinstance(practice_data,dict) else len(practice_data))
if isinstance(practice_data,list):
    scenes=practice_data
elif "scenes" in practice_data:
    scenes=practice_data["scenes"]
else:
    scenes=list(practice_data.values())
print("scene example",str(scenes[0])[:700])
import random
dev_ids={r["id"] for r in r1}
for mode in ["python","numpy"]:
    ix=(random.Random(7).sample(range(len(scenes)),64) if mode=="python"
        else np.random.default_rng(7).choice(len(scenes),64,replace=False))
    ids={scenes[int(i)]["id"] for i in ix}
    print("seed7 sample method",mode,"overlap with discovery",len(ids&dev_ids),"/",len(dev_ids))
with (W/"notes.md").open("a") as f:
    f.write("\nPass33 contact-gain fixed-input action effects:\n")
    for lab,rs in sg.items():
        f.write(lab+" "+str({k:round(float(np.mean([r[k] for r in rs])),4) for k in rs[0]})+"\n")
```
