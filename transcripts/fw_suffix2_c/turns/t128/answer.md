Across all 512 tagged episodes, the edit adds only one success, which does not meet the frozen gain criterion. I’ll record the exact uncertainty and continue the planned clean-instruction checks to measure possible collateral damage before reverting it.
```python
audit_rollout(43)
raw_final = paired_stats([40, 41, 42, 43])
s43 = paired_stats([43])
raw_gain_passed = raw_final["ci95_pp"][0] > 0
print("D", json.dumps(s43))
print("FINAL_TAGGED_VALIDATION", json.dumps(raw_final))
print("PREDECLARED_TAGGED_GAIN_CRITERION_PASSED", raw_gain_passed)

validation_results = {
    "candidate": "frozen_video_value_half",
    "tagged": raw_final,
    "tagged_gain_criterion_passed": raw_gain_passed,
    "status": "Complete planned safety checks; revert if retention criteria fail."
}
(root / "validation_results.json").write_text(
    json.dumps(validation_results, indent=2))
with (root / "notes.md").open("a") as f:
    f.write("\nFrozen validation raw D/pass43: " + json.dumps(s43) +
            ". Final raw A-D: " + json.dumps(raw_final) +
            ". Predeclared tagged gain criterion FAILED. Candidate will be "
            "reverted after completing planned strip/language checks; no tuning "
            "or replacement chosen from validation results.\n")

req44 = json.loads((root / "request44.json").read_text())
req40 = json.loads((root / "request40.json").read_text())
assert req44["episodes"] == req40["episodes"]
assert req44["model"] == "both" and req44["language"] == "strip"
print("REQUEST44")
print(json.dumps(req44, separators=(",", ":")))
```
