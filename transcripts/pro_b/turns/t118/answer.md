Batch 2 regressed by eight episodes, including object and semantic tasks. That is evidence against retaining the edit, but I will complete the prespecified test rather than make a decision from selected batches. No weights or test episodes will change.
```python
with (W/"notes.md").open("a") as f:
    f.write("\n## Final confirmation batch2 (pass37)\n"
            "82/128->74/128 (-6.2 +/-2.1pp),0wins/8losses. Losses in all5ptypes, including object-1/24 and semantic-2/28. "
            "Cumulative163/256->158/256. Strong negative evidence; complete the frozen plan, no retuning or alternative selection.\n")
print((W/"final_confirmation_request_3.json").read_text())
```
