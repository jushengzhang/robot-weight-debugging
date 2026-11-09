Batch B gained three successes, offsetting batch A’s loss; the pooled clean point estimate is now zero. I’ll save the paired uncertainty and verify that the next clean batch matches tagged batch C exactly.
```python
clean_ab = record_validation(45, "strip_B", [44, 45], "clean")
req46 = json.loads((root / "request46.json").read_text())
req42_saved = json.loads((root / "request42.json").read_text())
assert req46["episodes"] == req42_saved["episodes"]
assert req46["model"] == "both" and req46["language"] == "strip"
assert len(req46["episodes"]) == 128
print("NEXT_PASS", 46, "split=validation_c", "language=strip",
      "same_variants_and_seeds_as_pass42=True")
```
