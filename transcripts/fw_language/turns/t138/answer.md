The first seed-19 language batch gained two net successes, again smaller than its paired standard error. I will log it and load the last batch; the final language estimate will combine both seeds with uncertainty clustered by variant.
```python
verify_frozen_candidate()
rows39 = read_roll(39)
for model in ("original", "edited"):
    assert episode_keys([r for r in rows39 if r["model"] == model]) == episode_keys(replication_batches[0])
assert {r["category"] for r in rows39} == {"Language Instructions"}

result39 = paired_summary([39])
ledger["language"]["19"] = result39
ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
with (root / "notes.md").open("a") as f:
    f.write("\n## Pass 39: language replication seed19, batch 1\n"
            + "- " + json.dumps(result39) + "\n"
            + "- Gain is below paired SE. Candidate remains frozen for the last predeclared batch.\n")
print("Pass 39:", json.dumps(result39))
emit_request(40, replication_batches[1])
```
