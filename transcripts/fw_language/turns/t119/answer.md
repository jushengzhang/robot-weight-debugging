The pooled initial-state change is -3.1 points across 256 variants, exceeding the roughly 2-point guardrail, though uncertainty remains. The candidate currently fails retention criteria. I will finish the predeclared tests to document its effects, not tune another edit to compensate.
```python
ledger = record_safety_and_prepare_next(30)
with (work / "notes.md").open("a") as f:
    f.write("\n## Interim retention assessment after pass 30\n"
            "- Language seed7: -0.81 pp +/-1.72 SE, no independent gain.\n"
            "- Robot Initial States: observed -3.125 pp over 256 paired variants, "
            "above the allowed harm scale. Candidate currently fails retention criteria.\n"
            "- Continue the frozen evaluation solely for complete evidence. "
            "Do not compensate with further weight edits. Original remains the default final choice.\n")
```
