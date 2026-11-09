The final choice is the original checkpoint: neither candidate met the independent-validation requirement. I must remove the temporary audit edits so the submitted files agree with that decision.
```python
# Final submission cleanup: no further forward passes or candidate selection.
for name in list(edited()):
    drop_edit(name)
assert edited() == []
assert not list((W / "edited").rglob("*.npy"))
print("FINAL SUBMISSION: original checkpoint; zero edited tensors.")

audit50 = summarize_probe(50)
(W / "final_I065_routing_audit.json").write_text(json.dumps(audit50, indent=2))

final_report = {
    "decision": "original_checkpoint_no_edits",
    "reason": "Neither frozen candidate improved pooled independent validation.",
    "edited_tensors": edited(),
    "forward_passes_used": 50,
    "H15": {
        "pooled_validation": paired_summary(read_roll(26) + read_roll(45)),
        "prefail_preservation": paired_summary(read_roll(48)),
    },
    "I065": {
        "pooled_validation": paired_summary(read_roll(30) + read_roll(46)),
        "prefail_preservation": paired_summary(read_roll(47)),
    },
    "limitations": [
        "No validated recovery improvement was established.",
        "Pre-fail preservation is not an ordinary-LIBERO evaluation.",
        "Final four-input routing audits are mechanistic checks, not efficacy estimates.",
    ],
}
(W / "final_report.json").write_text(json.dumps(final_report, indent=2))
print(json.dumps(final_report, indent=2))

with (W / "notes.md").open("a") as f:
    f.write(
        "\n## FINAL DECISION after50 forward passes\n"
        "All temporary audit edits removed; edited()==[]. Submit original checkpoint.\n"
        "H15 pooled independent validation:133/254->133/254; "
        "I065 pooled independent validation:133/254->133/254. "
        "Neither meets the retention criterion. No validated recovery repair found.\n"
        "Pass50 same-input I065 routing audit saved as final_I065_routing_audit.json. "
        "Full decision and paired statistics saved as final_report.json. "
        "Unchanged weights preserve baseline ordinary behavior by construction; "
        "no ordinary-reset benchmark was available.\n"
    )
```