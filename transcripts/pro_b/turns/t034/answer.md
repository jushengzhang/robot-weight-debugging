The fixed-patch head attenuation also failed to improve task selection and damaged two object cases, so those heads are not a justified repair target. I’ll stop treating attention mass as the main explanation and inspect whether particular MLP updates suppress instruction-dependent differences or inject environment-dependent features into the residual stream.

```python
print("two-head intervention:",paired_summary(read_roll(12)))
clear_edits()
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 12: fixed-patch head attenuation rejected\n"
            "60/128->61/128 (+0.8 +/-2.6pp SE), task20/68 unchanged, "
            "object14/15->12/15. Fixed attention alone is not evidence of a harmful route. Reverted.\n"
            "Next: inspect MLP residual updates on matched initial observations for "
            "instruction-contrast cancellation and environment-specific feature injection.\n")
print("current edits:",edited())
```
