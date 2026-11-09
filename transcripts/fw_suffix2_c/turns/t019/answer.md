I will compare the video-only intervention with the original clean actions, then swap the same intervention to the action expert. These two tests separate the direct text-to-action route from the text-to-video-to-action route without changing both at once.
```python
m6=screen_metrics(6)
clear_edits()
text_dose("action",1.0)
print("NEXT CAUSAL EDIT:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\nPass6 intervention was video-only full text-specific ablation, proprio span preserved. "
            "Now action-only identical ablation for pass7; video restored to original.\n")
```
