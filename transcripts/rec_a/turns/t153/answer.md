Doubling the opening feedback caused five losses and no rescues, so that direction is rejected. I will use the remaining five passes for a fixed opposite-sign test: strengthen closing along the same visual route, then probe it and evaluate all three untouched validation panels without further tuning. The edit will be retained only if independent success gains clearly exceed their uncertainty; fewer flips alone will not qualify.
```python
Oneg=O17.clone(); Oneg[:,256:512]-=2*torch.outer(wg,dv)
save_edit(On,Oneg)
panels=json.loads((W/"independent_panels.json").read_text())
protocol={
    "candidate":"E17H1_visual_grip_retention_beta_minus2",
    "tensor":On, "beta":-2.0, "frozen_after_pass":45,
    "passes":{"46":"fixed64 discovery screen","47":"same16 paired mechanism probe",
              "48":"untouched validation panel2","49":"untouched validation panel3",
              "50":"untouched validation panel4"},
    "validation_episodes":sum(len(p) for p in panels[1:]),
    "decision":"Keep only if independent paired success gain exceeds 2 SE, loop failures improve, and solved-scene damage is small. Otherwise revert.",
    "ordinary_task_caveat":"Tool exposes only practice recovery scenes; original-task performance cannot be directly certified."
}
(W/"final_candidate_protocol.json").write_text(json.dumps(protocol,indent=2))
(W/"current_candidate.json").write_text(json.dumps({"name":protocol["candidate"],"beta":-2.0}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 45: opening-feedback dose4 rejected\n"
            "- 31/64 -> 26/64, -7.8 pp (SE 3.4), 0 rescues / 5 losses; loops 21 -> 26. Larger opening dose caused more damage.\n"
            "\n## Final five-pass precommitment\n"
            "- Reverse only the same rank-one E17H1 route (beta=-2), testing grip retention rather than increased reopening. The actual attended feature was not a reliable empty-hand detector; do not claim that semantic specificity.\n"
            "- Pass46 fixed discovery screen, pass47 paired mechanism probe, passes48-50 untouched validation panels2-4 (290 paired episodes), no further tuning.\n"
            "- Retain only with independent gain >2 paired SE, lower loop failures, and small damage to already-solved recovery scenes; otherwise submit no edits. Ordinary-task endpoint is unavailable.\n")
print("Protocol:",json.dumps(protocol,indent=2))
print("Validation panels:",[len(p) for p in panels[1:]])
```
