The geometric query edit did not reduce the unwanted word attention, so I will discard it rather than claim a successful intervention. I will instead halve the query scale of this one head, a direct way to soften its extreme word fixation while retaining the head’s output pathway; a paired rollout will determine whether that redistribution helps.
```python
for n in list(edited()): drop_edit(n)
qname=f"{expert}layers.16.self_attn.q_proj.weight"
qw=load_original(qname)
qw[768:1024]*=0.5
save_edit(qname,qw)
(W/"current_candidate.json").write_text(json.dumps({"name":"E16H3_Q_0.5","layer":16,"head":3,"query_gain":0.5}))
with (W/"notes.md").open("a") as f:
    f.write("- Pass 31 geometric query edit barely changed looping word attention and reduced visual attention in one success; rejected without rollout.\n"
            "- Next causal candidate: E16H3 query-block gain 0.5 only, to soften this head's word fixation without removing its output pathway.\n")
print("Current edits",edited())
```
