I will compare the combined gate with the action-only gate. I will keep this direction only if it improves tagged-to-clean action agreement without appreciable clean-language drift.
```python
m25=probe_metrics(25)
print(json.dumps(m25,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 25: both-expert context gate strength6 threshold.5\n"+json.dumps(m25)+"\n")
(W/"candidate_both_gate.json").write_text((W/"current_edit.json").read_text())
print("Action-only vs both gate gap ratios:",m24["gap_ratio"],m25["gap_ratio"])
```
