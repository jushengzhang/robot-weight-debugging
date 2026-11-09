Language batch B is tied, leaving a pooled gain of three successes out of 256. I’ll log the estimate and prepare the final reserved batch, completing the planned validation budget.
```python
record_validation(49, "language_B", [48, 49], "language")
assert all(r["category"] == "Language Instructions"
           for r in audit_rollout(49))
show_next_request(50)
```
