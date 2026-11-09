Batch 1 is positive but still within sampling uncertainty. I will record it and load the next preselected batch without changing the edit or the acceptance rule.
```python
with (W/"notes.md").open("a") as f:
    f.write("\n## Final confirmation batch1 (pass36)\n"
            "81/128->84/128 (+2.3 +/-2.3pp),6wins/3losses. "
            "Environment+1,position+1,task+1; object/semantic unchanged. "
            "Still inconclusive; continue all five frozen batches.\n")
print((W/"final_confirmation_request_2.json").read_text())
```
