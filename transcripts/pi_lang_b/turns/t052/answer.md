The reserved validation is complete. Before auditing the other categories, I will total its results and collect only execution-error cases for a retry. I will also prepare controlled replays to compare the compact artifact with an exact-source artifact that changes no untargeted embedding values.
```python
vp=[p for k in range(18,22) for p in pair_roll(k)]
vc=[p for p in vp if is_complete(p)]
print("512-variant validation raw:",paired_stats(vp))
print("Completed validation:",paired_stats(vc))
for suite in sorted({p["original"]["suite"] for p in vp}):
    print(suite, paired_stats([p for p in vc if p["original"]["suite"]==suite]))
print("Batch21 completed discordances:")
for p in pair_roll(21):
    a,b=p["original"],p["edited"]
    if is_complete(p) and bool(a["success"])!=bool(b["success"]):
        print("FIX" if b["success"] else "REG",a["suite"],a["id"],repr(a["instruction"]))
retry_eps=[{"suite":p["original"]["suite"],"id":p["original"]["id"],"seed":p["original"]["seed"]}
           for p in vp if not is_complete(p)]
(W/"validation_error_retry.json").write_text(json.dumps(
    {"type":"rollout","model":"both","episodes":retry_eps},indent=1))
print("Validation pairs needing infrastructure retry:",len(retry_eps))

print("\nEasy goal controls:")
for r in read_roll(2):
    print(r["id"],r["success"],repr(r["instruction"]))
pr11=json.loads((W/"probe_11"/"results.json").read_text())
probe_inputs=list(dict.fromkeys(r["input"] for r in pr11))
print("\nExisting probe input count",len(probe_inputs))
for i,p in enumerate(probe_inputs):
    print(i,Path(p).parent.name,Path(p).name)
x=np.load(W/"probe_11"/"probe_orig_0_acts.npz")
print("Activation schema:",[(k,x[k].shape) for k in x.files[:12]])
print("Previous probe layers:",x.files)
# Use the existing diagnostic replay set, never validation examples, for storage-equivalence probes.
req={"type":"probe","model":"both","inputs":probe_inputs,
     "return":{"layers":[0,3,6,9,12,15,17],"vlm_layers":[0,3,6,9,12,15,17],"what":"resid"}}
(W/"mechanism_probe_request.json").write_text(json.dumps(req,indent=1))
print("\nNEXT PROBE:")
print(json.dumps(req,separators=(",",":")))
with (W/"notes.md").open("a") as f:
    f.write("\nFrozen seed23 reserved validation complete: raw "+str({k:v for k,v in paired_stats(vp).items() if k!="errors"})
            +"; completed "+str({k:v for k,v in paired_stats(vc).items() if k!="errors"})
            +f". {len(retry_eps)} pairs have infrastructure errors and are queued for same-seed retry. No new rows chosen from validation.\n")
```
