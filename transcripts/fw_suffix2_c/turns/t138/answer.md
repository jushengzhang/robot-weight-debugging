Language batch A gained three successes. I’ll record its paired estimate and load batch B, keeping the remaining paraphrase checks separate from the already-completed tagged-efficacy test.
```python
record_validation(48, "language_A", [48], "language")
assert all(r["category"] == "Language Instructions"
           for r in audit_rollout(48))
show_next_request(49)
```
