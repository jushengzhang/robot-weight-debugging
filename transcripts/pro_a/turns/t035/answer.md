The positional edit is the first candidate with gains on the weak perturbations: +3.9 ± 2.0 points overall, including two additional task successes and one environment success, with only one lost success. This is still development evidence. I will test a stronger dose and inspect whether it actually reduces reliance on the exceptional position features.
```python
r12=read_jsonl(W/"roll_12.jsonl")
report_pairs(r12)
print("Changed outcomes under position-register retention50%:")
for key,a in paired_records(r12).items():
    if a["original"]["success"]!=a["edited"]["success"]:
        o,e=a["original"],a["edited"]
        print(key,o["ptype"],"win" if e["success"] else "loss",o["task"],
              "flips",o["grip_flips"],e["grip_flips"],
              "old_goal",o["orig_goal_reached"],e["orig_goal_reached"])
snapshot("provisional_register050")
clear_edits()
set_position_outlier_retention(0.2)
snapshot("candidate_position_register_keep020")
print("Stronger-dose row norms:",[(i,round(float(load(pe_name)[i].norm()),2)) for i in outlier_positions])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 12: candidate F, positional outlier residual retain50%\n"
            "dev128 69->74, +3.9 +/-2.0pp SE, 6 wins/1 loss. "
            "Task5->7, environment17->18, position3 unchanged, object23 unchanged, semantic21->23; "
            "loops29->24, old-goal failures13->12. Provisional only, not independent validation. "
            "Saved provisional_register050. Next dose retain20%, same five rows, no other edits.\n")
```
