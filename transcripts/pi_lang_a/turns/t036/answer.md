Changing only the common embedding component gives just one rescue, compared with ten for complete row substitution. I will now test the complementary directions while preserving each source row’s common component. This will show whether the useful signal is mainly token-specific rather than a generic magnitude adjustment.

```python
print("common-only valid:",paired_summary(13))
for key,a,b in paired_records(13):
    if a["success"]!=b["success"]:
        print("changed",key,a["instruction"])
apply_component_control(candidate_full_meta,"orthogonal","orthogonal_direction_control")
with (W/"notes.md").open("a") as f:
    f.write("\nPass13 common-direction-only control: "
            f"{paired_summary(13)} (48 requested). Only1 rescue vs10 for complete substitutions. "
            "Now testing orthogonal complement of the common embedding mean direction.\n")
print("Current edit: token-specific delta components, shared mean component preserved.")
```
