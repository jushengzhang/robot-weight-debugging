The 50% illumination-component edit has the predicted direction but is not yet statistically convincing: +2.3 points, paired SE 2.1, with fewer loop failures. I will test full removal of the same single component to establish a dose-response before deciding whether this mechanism is worth validating.
```python
set_dc_edit(1.0)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass12: D at0.5\nDevelopment71->74/128 +2.3pp SE2.1; env14->16, position4->5, task8->9, object18->18, semantic27->26. Loop failures30->26. Directionally promising but inconclusive. Test dose1.0 next.\n")
```
