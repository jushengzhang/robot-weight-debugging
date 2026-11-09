The maps identify a concrete instruction-reading pathway at expert layer 4. I will strengthen only heads 3 and 7 by 50% through their output-projection column blocks, leaving visual heads and every other layer unchanged. This tests whether the instruction is present but underweighted relative to scene-conditioned signals.
```python
set_head_edit({4:{3:1.5,7:1.5}})
print("Candidate C1:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass10 and candidate C1\nEarly expert L4 h0/h4 base+BOS dominated; h3/h7 allocate23-30% to content words vs1-3% h0/h4. h3 reads cream/cheese, h7 broader syntax/destination. Candidate C1: only expert L4 O-projection head blocks3 and7 x1.5. All previous edits removed.\n")
```
