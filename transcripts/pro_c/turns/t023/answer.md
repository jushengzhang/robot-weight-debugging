The selective key edit is better than the BOS-row edit, but +3.1 points with a 2.2-point paired standard error is still inconclusive and does not fix task selection. I will test the full-removal dose on the same development set; this will reveal whether stronger sink suppression helps or whether the sink is not the main failure mechanism.
```python
set_key_sink_edit(1.0)
print("Candidate B dose: columns 604/738 set to zero in VLM K at",key_layers)
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 7-8: candidate B at0.5\nProbe BOS mass roughly halves at L6/8, words increase without uniform gripper shift. Paired development n128:71->75, +3.1pp SE2.2; env14->18, object18->19, position4->4, semantic27->27, task8->7. Inconclusive, no task repair. Next dose1.0 (zero only selected columns).\n")
```
