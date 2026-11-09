The 384-variant validation total is negative, so I will reject and revert the query-softening edit. I will preserve the final 247 variants as untouched confirmation data and use the remaining discovery budget to examine whether paraphrases weaken useful instruction signals, rather than create excessive attention sharpness.
```python
validation_summary = paired_summary(paired_records([17, 18, 19]))
print("Completed independent validation:", validation_summary)
(work / "L11H19_query075_validation_summary.json").write_text(
    json.dumps(validation_summary, indent=2))
for n in list(edited()):
    drop_edit(n)
assert not list(edited())

with (work / "notes.md").open("a") as f:
    f.write("\n## Pass 19: query-softening candidate REJECTED\n"
            "- Three predeclared validation batches completed: " +
            json.dumps(validation_summary) + "\n"
            "- All edits reverted. No demonstrated language improvement.\n"
            "- Confirmation batches 4-5 (247 variants) remain entirely unrun. "
            "Do not tune on validation episodes or inspect their detailed failure patterns.\n"
            "- Remaining 21 passes: allow at most five renewed discovery/causal passes "
            "using the original discovery data; reserve two for untouched language "
            "confirmation, twelve for six-category checks, and two for seed replication. "
            "Any new candidate must pass independent confirmation; otherwise submit original.\n")

print("Active edits:", list(edited()))
print("Small diagnostic files:")
for p in sorted(work.iterdir()):
    if p.is_file() and p.suffix in (".json", ".pt", ".npy") and not p.name.startswith("roll_"):
        print(p.name, p.stat().st_size)
r3 = read_jsonl(work / "roll_3.jsonl")
print("Discovery record keys:", list(r3[0]))
print("Discovery input directory listing:",
      [p.name for p in (work / "roll_3_frames").iterdir()][:12])
print("Probe directories:", sorted(p.name for p in work.glob("probe_*") if p.is_dir()))
```
