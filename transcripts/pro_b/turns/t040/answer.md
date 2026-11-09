The eight smooth spatial modes account for about 35% of ordinary positional-embedding variation. The quarter-dose edit leaves all five high-norm rows exactly intact. I’ll expand the diagnostic panel with additional position perturbations so that the causal test is not driven by just one or two moved-object successes.

```python
additional_position=stratified_pick([e for e in partitions["diagnostic"]
                                     if e["id"] not in {d["id"] for d in diag_eps}],
                                    {"position":53},"position-causal-extra")
pos_test_eps=diag_eps+additional_position
print("extra position IDs:",[e["id"] for e in additional_position])
(W/"position_test_ids.json").write_text(json.dumps([e["id"] for e in pos_test_eps]))
# Show where position failures move a distractor while leaving the instructed object still.
for eid in (1496,1637,1276,4249,4625,1012):
    r=next(r for r in r1 if r["id"]==eid)
    moved=sorted(r["obj_moved_m"].items(),key=lambda kv:-kv[1])
    print(eid,r["instruction"],"moved:",[(k,round(v,3)) for k,v in moved[:3]])
```
