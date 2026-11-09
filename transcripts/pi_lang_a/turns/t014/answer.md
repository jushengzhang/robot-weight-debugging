I will measure the action and representation changes from the half-dose edit and list every diagnostic instruction containing either edited token. That list will define a small causal rollout test without selecting episodes based on the edited model’s outcomes.

```python
p4=json.loads((W/"probe_4/results.json").read_text())
print("entry:",str(p4[0])[:1800])
for x in p4:
    if x["idx"]<2:
        print(x["idx"],x["arm"],np.round(np.load(x["action"])[:5].mean(0),4))
a4=np.load(p4[0]["activations"])
print("activation keys/shapes:",[(k,a4[k].shape) for k in a4.files][:30])
def get_probe_entry(result, idx, arm):
    return next(x for x in result if x["idx"]==idx and x["arm"]==arm)
def action_dist(a,b):
    return float(np.sqrt(np.mean((a-b)**2)))
ref=np.load(get_probe_entry(p4,1,"orig")["action"])
for arm in ["orig","edit"]:
    pr=get_probe_entry(p4,0,arm)
    act=np.load(pr["action"])
    print("basin-to-reference action RMS",arm,round(action_dist(act,ref),5))
for i in [1,2,3]:
    a=np.load(get_probe_entry(p4,i,"orig")["action"])
    b=np.load(get_probe_entry(p4,i,"edit")["action"])
    print("unaffected input",i,"max difference:",float(np.max(np.abs(a-b))))
pair_diag_records=[r for r in diag_records if any(word_ids(s)[0] in sp.encode(r["instruction"],out_type=int) for s in alias_pair)]
print("pair token exposures:",len(pair_diag_records))
for r in pair_diag_records:
    print(r["suite"],r["id"],r["success"],r["instruction"])
```
