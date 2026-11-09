Half-retention gives +2.3 points on discovery, but its 2.6-point standard error is larger than the gain. I will complete the predeclared dose test by removing this single mode entirely, while preserving the rest of the head.
```python
s23 = paired_summary(paired_records([23]))
assert {(k[0], k[1]) for k in paired_records([23])} == discovery_keys
torch.save({"name": text_name, "tensor": load(text_name), "retention": 0.5},
           work / "candidate_action_L11H5_cross_O_half.pt")
print("Half-retention discovery:", s23)
print("Stronger dose:", set_text_mode_retention(0.0))
with (work / "notes.md").open("a") as f:
    f.write("\n## Pass 23: direct text-mode half-retention discovery rollout\n"
            "- " + json.dumps(s23) + "\n"
            "- Positive estimate is smaller than its SE; not accepted. "
            "Original within-pass baseline 88/128 differs from earlier repeats, "
            "so do not compare unpaired totals across passes.\n"
            "- Saved half-retention candidate; apply zero retention of the same "
            "single mode for the final predeclared exploratory dose test.\n")
```
