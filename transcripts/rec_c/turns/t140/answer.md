I will separate opening changes on empty grasps from changes on contact controls. That distinction is essential: reopening an empty grasp is the intended effect, whereas reopening a valid grasp could drop the object.
```python
a46 = probe_actions(46)
d46 = a46["edit"]-a46["orig"]
res46={}
for group in dict.fromkeys(group_key(r) for r in m):
    inds=[i for i,r in enumerate(m) if group_key(r)==group]
    o,e=a46["orig"][inds,:5],a46["edit"][inds,:5]
    d=e-o
    res46[group]={
        "n":len(inds),
        "dz":round(float(d[:,:,2].mean()),4),
        "xy_RMS":round(float(np.sqrt(np.mean(d[:,:,:2]**2))),4),
        "rot_RMS":round(float(np.sqrt(np.mean(d[:,:,3:6]**2))),4),
        "dgrip":round(float(d[:,:,6].mean()),4),
        "orig_close_frac":round(float(np.mean(o[:,:,6]>0)),4),
        "edit_close_frac":round(float(np.mean(e[:,:,6]>0)),4),
        "close_to_open":round(float(np.mean((o[:,:,6]>0)&(e[:,:,6]<=0))),4),
        "open_to_close":round(float(np.mean((o[:,:,6]<=0)&(e[:,:,6]>0))),4)
    }
    print(group,res46[group])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass46: empty-grasp reopening+clearance fixed-input probe\n")
    for k,v in res46.items():
        f.write(k+" "+json.dumps(v)+"\n")
# Prepare outcome-independent remaining evaluation pools.
seen2=set()
for p in W.glob("roll_*.jsonl"):
    for line in p.read_text().splitlines():
        if line.strip():
            seen2.add(json.loads(line)["id"])
diagnostic_ids={r["id"] for r in m if "id" in r}
prefail_ids=sorted(r["id"] for r in scenes if r["stage"]=="pre-fail")
rng=np.random.default_rng(6047)
fresh_ids=[r["id"] for r in scenes if r["id"] not in (seen2|diagnostic_ids) and r["stage"]!="pre-fail"]
rng.shuffle(fresh_ids)
frozen_final_pools={"safety_prefail":prefail_ids,
                    "test47_fresh":fresh_ids[:48],
                    "test48_fresh":fresh_ids[48:112],
                    "test50_fresh":fresh_ids[112:176]}
(W/"frozen_final_pools.json").write_text(json.dumps(frozen_final_pools))
print("pre-fail proxy IDs",prefail_ids)
print("test47 fresh IDs",frozen_final_pools["test47_fresh"])
print("fresh remaining",len(fresh_ids))
```
