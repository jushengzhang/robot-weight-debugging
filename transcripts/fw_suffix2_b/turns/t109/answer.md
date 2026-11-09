This clean batch showed a substantial loss: 15 fewer successes, or -11.7 points. I’ll verify that the weights still match the frozen candidate and keep this result fully included. The pooled clean estimate is now about -2.0 points, so safety is a serious concern—not something the tagged gains can override.
```python
frozen_hashes = json.loads((W/"frozen_manifest.json").read_text())
assert set(edited()) == set(frozen_hashes)
bad = [n for n,h in frozen_hashes.items()
       if file_sha(W/"edited"/(n+".npy")) != h]
assert not bad, bad
print("All 308 frozen weight hashes still match.")
with (W/"notes.md").open("a") as f:
    f.write("\nPass42 important safety warning: clean batch3 original87/128 vsedited72/128, "
            "-11.71875pp (4gains/19losses). ALL308 current weight hashes reverified "
            "against frozen_manifest.json after this result; no edit drift or tuning. "
            "Include this batch unconditionally. Continue planned validation, and revert "
            "if final safety or gain criteria fail.\n")
status = validation_status()
```
