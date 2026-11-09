The earlier attempts suggest that erasing average tag directions is not enough, and that some selective gates actually move actions farther from the clean reference. I’ll read the remaining conclusions and recover the latest readout code so I can assess whether finer context measurements are worth the pass budget.
```python
text = (W / "prior_notes.md").read_text()
print("PRIOR NOTES, REMAINDER:\n" + text[11500:])
print("\nWORK FILES:")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    elif not p.name.startswith(("roll_", "probe_", "prior_candidate")):
        print(p.name + "/", len(list(p.iterdir())))
print("\nCURRENT EDITS:", len(edited()), list(edited())[:20])
print("\nPYTHON FILES:", [p.name for p in W.glob("*.py")])
```
