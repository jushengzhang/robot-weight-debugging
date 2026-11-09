The independent rollout is neutral: 9 wins and 9 losses. The probe identified a concrete defect in the intervention itself—normal contact and open-gripper states receive unwanted downward motion. I will investigate that leakage using actual attention-weighted features, then test whether centering the feature at the attention readout removes it. This is a revision, so the previous validation batch will no longer count as fresh evidence.
```python
normal_source_indices=[2*j+1 for j in range(22) if j in source_indices]+list(range(44,60))
c0=float(X35["wrist"][normal_source_indices,674].mean())
ref=torch.zeros(2048);ref[674]=1.0/c0
print("normal sink reference mean",round(c0,4),
      "reference normalized scores range",
      round(float((X35["wrist"]@ref).min()),3),round(float((X35["wrist"]@ref).max()),3))
torch.save({"reference_feature":ref,"normal_source_indices":normal_source_indices,
            "nominal":0.60,"feature_direction":d,
            "value_direction":dv},W/"clearance_centering_design.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass40: fresh clearance0.60 validation neutral\n"
            "126 valid pairs:77 vs77,9 wins/9 losses (0.0+/-3.4pp); loops40 vs42, "
            "mean flips10.22 vs11.87; libero_10 loses4 successes. Not a validated repair.\n"
            "Known intervention defect from probes: downward leakage in open/contact controls. "
            "Next diagnose actual attention-weighted feature via expert17 attn_out difference. "
            "Use a normalized channel674 reference to center this feature at the attention readout, "
            "not at an unweighted token mean. Centering source is14 source contacts+16 open controls; "
            "no success labels or reserved-contact outcomes used. confirmation_batch3 still unseen.\n")
```
