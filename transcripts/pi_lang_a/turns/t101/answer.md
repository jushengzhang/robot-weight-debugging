The first hard-language batch gained 8 successes with no regressions, including 7 gains at difficulty 5. I will audit and save it, then run the remaining 102 variants.
```python
def log_fixed_check_and_next(k, next_k, label):
    audit = audit_request(k)
    stats = paired_stats(read_roll(k))
    print(label + ":", json.dumps({**audit, **stats}, separators=(",", ":")))
    notes = (W / "notes.md").read_text()
    notes = re.sub(r"Forward passes used: \d+/50\.", f"Forward passes used: {k}/50.", notes)
    notes += f"\nPass{k} {label}: {json.dumps(stats)}. Manifest exact; frozen18 unchanged.\n"
    (W / "notes.md").write_text(notes)
    print(f"REQUEST{next_k}")
    print((W / f"request{next_k}.json").read_text())

log_fixed_check_and_next(47, 48, "Third-seed D4/D5 check, first half")
```
