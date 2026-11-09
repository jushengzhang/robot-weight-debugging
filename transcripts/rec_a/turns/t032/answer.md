The gripper's dependence on Gaussian noise is weak, while its dependence on visible finger width is strong in both successes and failures. I will test the late finger-focused head next. Unlike the earlier camera-routing edits, this targets the route that can reinforce “keep the current grasp state” even after an unsuccessful grasp.
```python
cn=head_o_edit(15,2,0.5)
print("Diagnostic edit:",cn,"only head-2 columns 512:768 at half strength")
print("Relative tensor change",round(float((load(cn)-load_original(cn)).norm()/load_original(cn).norm()),5))
with (W/"notes.md").open("a") as f:
    f.write("\n## Observational gripper check\n"
            "- Across saved observations: Gaussian gripper-coordinate/output correlations "
            "0.093 successes, 0.096 loops; finger-width/output correlations -0.567 successes, -0.636 loops.\n"
            "- This supports strong visible-state dependence, not a globally noise-dominated gripper.\n"
            "- Next localization: late finger-focused expert L15 H2 O block at 0.5.\n")
```
