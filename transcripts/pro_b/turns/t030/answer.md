I’ll measure whether this edit acts differently from the rejected mid-layer key/value edits, and then use rollouts to test whether any routing change actually corrects the chosen object or goal.

```python
pr10=json.loads((W/"probe_10"/"results.json").read_text())
by10=collections.defaultdict(dict)
for r in pr10: by10[r["idx"]][r["arm"]]=r
for i,rs in by10.items():
    a=np.load(workpath(rs["orig"]["action"])); b=np.load(workpath(rs["edit"]["action"]))
    print("input",i,"action RMS",round(float(np.sqrt(((a-b)**2).mean())),4),
          "xyz mean delta",[round(float(x),4) for x in (b-a).mean(0)[:3]])
    for l in (3,6,12,17):
        m0=attention_metrics(rs["orig"],l); m1=attention_metrics(rs["edit"],l)
        print(" L",l,"content %",round(m0[3]*100,2),"->",round(m1[3]*100,2),
              "BOS %",round(m0[2]*100,2),"->",round(m1[2]*100,2))
# Additional diagnostic task episodes, never from validation/confirmation.
additional_task=stratified_pick([e for e in partitions["diagnostic"]
                                 if e["id"] not in {d["id"] for d in diag_eps}],
                                {"task":53},"task-causal-extra")
lex_test_eps=diag_eps+additional_task
print("extra task IDs:",[e["id"] for e in additional_task])
(W/"lex_test_ids.json").write_text(json.dumps([e["id"] for e in lex_test_eps]))
```
