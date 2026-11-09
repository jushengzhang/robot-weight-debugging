Across the full third-seed hard-language check, the edit gained 14 successes: 16 rescues and 2 regressions. I will save that secondary result, then run the precommitted fresh lighting sample to increase safety-check coverage.
```python
print("Pass48 audit:", audit_request(48))
hard_rows = read_roll(47) + read_roll(48)
hard_summary = {
    "overall": paired_stats(hard_rows),
    "by_difficulty": grouped_stats(hard_rows, "difficulty"),
    "by_suite": grouped_stats(hard_rows, "suite")
}
hard_summary["exact_p"] = exact_discordance_p(hard_summary["overall"])
(W / "final18_hard_third_seed_summary.json").write_text(json.dumps(hard_summary, indent=2))
print("Third-seed hard-language:", json.dumps(hard_summary, separators=(",", ":")))

notes = (W / "notes.md").read_text()
notes = re.sub(r"Forward passes used: \d+/50\.", "Forward passes used: 48/50.", notes)
notes += ("\nPass48 completes all204 hard-language third-seed pairs: " +
          json.dumps(hard_summary["overall"]) + ". By difficulty: " +
          json.dumps(hard_summary["by_difficulty"]) +
          ". This is a secondary difficulty-enriched check, not pooled into the "
          "primary1220-pair estimate. Frozen18 unchanged. Final two passes are safety only.\n")
(W / "notes.md").write_text(notes)
print("REQUEST49")
print((W / "request49.json").read_text())
```
