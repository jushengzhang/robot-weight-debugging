Clean batch C gained seven successes. I’ll add that result to the pooled estimate and verify the last clean batch. It does not change the failed tagged-efficacy result or the predeclared retention rule.
```python
clean_abc = record_validation(46, "strip_C", [44, 45, 46], "clean")
req47 = json.loads((root / "request47.json").read_text())
req43_saved = json.loads((root / "request43.json").read_text())
assert req47["episodes"] == req43_saved["episodes"]
assert req47["model"] == "both" and req47["language"] == "strip"
assert len(req47["episodes"]) == 128
print("NEXT_PASS", 47, "split=validation_d", "language=strip",
      "same_variants_and_seeds_as_pass43=True")
```
