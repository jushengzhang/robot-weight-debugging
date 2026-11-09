The first Objects Layout batch is neutral, with three rescues and three regressions. I will log it and load the second batch to complete the six-category safety evaluation.
```python
log_safety_pass(37)
assert safety_plan[11]["category"] == "Objects Layout" and safety_plan[11]["batch"] == 2
emit_request(38, safety_plan[11]["episodes"])
```
