Both finger-head attenuations delayed reopening in the same empty-closed grasp state; the late head delayed it by two action tokens while leaving most confident decisions unchanged. That is evidence in the opposite direction from my initial hypothesis: this visual route may help reject a failed grasp rather than cause the loop. I will test strengthening it, with the half-strength result serving as the negative dose.
```python
clear_edits()
cn=head_o_edit(15,2,2.0)
print("Candidate: L15 H2 output gain 2.0; relative tensor delta",
      round(float((load(cn)-load_original(cn)).norm()/load_original(cn).norm()),5))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 13: late finger head localization\n"
            "- L15 H2 half-output: 2/160 gripper signs changed, both in input 13 "
            "(scene 1269, initially empty closed gripper); reopening delayed from token 2 to token 4.\n"
            "- Other 15 inputs: action changes mostly ~0.001; downstream MLP responses partly cancel "
            "the head perturbation (L15 ~9-11%, L17 ~1-2%).\n"
            "- Both early and late finger-head attenuation delay reopening in this state. "
            "Next test reverses the intervention: L15 H2 gain 2.0, to strengthen visual failed-grasp feedback.\n")
```
