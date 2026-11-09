The routing edit also has zero pooled gain: its five validation-B rescues net out its five validation-A losses. Neither frozen candidate meets the retention criterion. I will use the final passes for preservation and execution controls, not further tuning. First I will test the current routing edit on the reserved pre-fail scenes, which are only a surrogate—not an ordinary LIBERO reset evaluation.
```python
print("H15 independent validation:",pooled_stats([26,45]))
print("I065 independent validation:",pooled_stats([30,46]))

prefail_ids=sorted(s["id"] for s in scenes if s["stage"]=="pre-fail")
prefail_episodes=[{"id":sid,"grip":g,"seed":seed}
                  for seed in [7,31,79] for sid in prefail_ids for g in ["open","close"]]
assert len(prefail_episodes)==96
(W/"prefail_preservation_episodes.json").write_text(json.dumps(prefail_episodes))
print("Pre-fail IDs:",prefail_ids,"seeds",[7,31,79],"episodes",len(prefail_episodes))

replication_summary={"H15":pooled_stats([26,45]),"I065":pooled_stats([30,46]),
                     "retained_candidate":None,
                     "reason":"Neither fixed candidate has a pooled independent-validation gain."}
(W/"replication_summary.json").write_text(json.dumps(replication_summary,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## I065 independent replication, pass46\n"
            "ValidationB60/126->65/126,+4.0+/-3.1pp,10 rescues/5 regressions. "
            "Pooled A+B133/254->133/254:zero net gain. Neither H15 nor I065 qualifies. "
            "Decision: retain original weights; remaining4 passes are preservation/control audits, "
            "not new parameter selection. Pre-fail scenes are only a surrogate for ordinary tasks; "
            "no ordinary-reset API exists.\n")
```
