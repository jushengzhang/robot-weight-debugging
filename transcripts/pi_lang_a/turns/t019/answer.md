Full substitution rescues five basin episodes versus two at half dose, but the rollout tool also reports some `null` terminations and the repeated original totals differ. I will inspect those records before interpreting the seed-level result, then isolate the basin row from the cupboard row to identify which change is actually responsible.

```python
p6=json.loads((W/"probe_6/results.json").read_text())
ref6=np.load(get_probe_entry(p6,1,"orig")["action"])
print("action dose response:")
print("zero",round(action_dist(np.load(get_probe_entry(p6,0,"orig")["action"]),ref6),5))
print("half",round(action_dist(np.load(get_probe_entry(p4,0,"edit")["action"]),ref6),5))
print("full",round(action_dist(np.load(get_probe_entry(p6,0,"edit")["action"]),ref6),5))
for k in [5,7]:
    rr=read_roll(k)
    print("\nroll",k,"records",len(rr),"keys",rr[0].keys())
    print("null example:",str(next((r for r in rr if not r.get("termination")),{}))[:1600])
    groups=defaultdict(list)
    for r in rr:
        groups[(r["suite"],r["id"],r["model"])].append(r)
    for key, rs in groups.items():
        print(key, "success",sum(bool(r["success"]) for r in rs),"/",len(rs),
              "terminations",dict(Counter(str(r.get("termination")) for r in rs)))
print("Basin original/common-row similarity:")
print("basin==rounded embedding?",torch.equal(E0[word_ids("basin")[0]],E0[word_ids("rounded")[0]]))
basin_only=apply_aliases({"basin":"bowl"},1.0,"basin_only")
with (W/"notes.md").open("a") as f:
    f.write("\nPass6 full-dose probe saved. Pass7 full-dose same32 pairs: "
            "original19/32 edited23/32 (+12.5pp,SE7.4;5 rescues1 regression), "
            "basin0/8->5/8. Null terminations and original long-suite totals vary across runs; "
            "do not treat repeat-baseline differences as weight effects. "
            "Next: single-row ablations.\n")
```
