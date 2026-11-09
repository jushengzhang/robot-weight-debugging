The jitter audit rules out that counterfactual: open and closed starts have different object displacements, even with the same reported seed. I will not infer a finger-specific feature from them. With the remaining budget, I will make one reverse-direction test of the existing layer-7 visual-variation edit; if it does not produce a convincing, independently validated gain, I will retain the original weights rather than submit a speculative repair.
```python
assert content_name == LP+"layers.7.self_attn.v_proj.weight"
v_reduced=set_content_edit(-.5)
print("Candidate N reverse dose: retain 50% of rank4 visual-variation component.")
print("Relative weight change:",round(float((v_reduced-v_orig).norm()/v_orig.norm()),4))
print("Current edit:",edited())

with (W/"notes.md").open("a") as f:
    f.write("\n## Important matched-grip confound\n"
            "Initial open/closed starts have DIFFERENT object jitters even for seed17. "
            "Zero pose+jitter-exact pairs among48 diagnostic scenes. "
            "Do not interpret open/closed differences as causal finger-only features. "
            "Paired original-vs-edited comparisons remain valid (same id/grip/seed). "
            "No gripper-image feature edit was constructed.\n"
            "\n## N reverse dose\n"
            "One final direction test: alpha=-.5, retain half the existing rank4 "
            "within-task visual-variation component at VLM7 V. "
            "Amplification hurt; attenuation tests overreaction to view variation, "
            "not a claim that the basis isolates fingers. Require independent validation.\n")
```
