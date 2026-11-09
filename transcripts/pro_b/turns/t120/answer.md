Batch 3 recovered four episodes, leaving the cumulative result nearly unchanged from baseline. I will record this and continue with the fourth fixed batch; the mixed results reinforce why the larger paired test is necessary.
```python
with (W/"notes.md").open("a") as f:
    f.write("\n## Final confirmation batch3 (pass38)\n"
            "70/128->74/128 (+3.1 +/-1.9pp),5wins/1loss. "
            "Environment-1,object+2,position+1,semantic+1,task+1. "
            "Cumulative233/384->232/384; no demonstrated overall gain.\n")
print((W/"final_confirmation_request_4.json").read_text())
```
