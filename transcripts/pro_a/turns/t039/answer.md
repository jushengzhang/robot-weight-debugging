The stronger dose increases task successes further, from 5 to 8, but loses gains elsewhere; position remains unchanged. This suggests a real but limited effect, not a general repair. I will localize it more tightly by changing only patch 193—the directly observed base-camera hotspot—while restoring the other four exceptional rows.
```python
r14=read_jsonl(W/"roll_14.jsonl")
report_pairs(r14)
print("Task outcomes by register dose (1.0 original, 0.5, 0.2):")
pp12=paired_records(r12); pp14=paired_records(r14)
for key in pp12:
    a=pp12[key]; b=pp14[key]
    if a["original"]["ptype"]=="task" and (a["edited"]["success"]!=a["original"]["success"]
                                        or b["edited"]["success"]!=b["original"]["success"]):
        print(key, a["original"]["task"],
              [a["original"]["success"],a["edited"]["success"],b["edited"]["success"]],
              "old_goal",[a["original"]["orig_goal_reached"],a["edited"]["orig_goal_reached"],b["edited"]["orig_goal_reached"]])
clear_edits()
set_position_outlier_retention(0.2,positions=[193])
snapshot("candidate_position193_keep020")
print("Changed rows:",(load(pe_name)!=pe0).any(1).nonzero().flatten().tolist())
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 13-14: positional residual retain20%, five outliers\n"
            "Probe: L9H4 base193 attention task1946 13.23->0.11%, position1896 28.90->0.40%; "
            "wrist211 largely preserved (77.57->72.50% task;77.36->75.17% position). "
            "Dev12869->71, +1.6 +/-2.7pp SE, 7 wins/5 losses. Task5->8, position3->3, "
            "environment17->16, object23->23, semantic21->21. "
            "Task dose-response 5(original),7(retain50%),8(retain20%), but stronger dose harms other types. "
            "Next isolate row193 retain20%, all other rows restored.\n")
```
