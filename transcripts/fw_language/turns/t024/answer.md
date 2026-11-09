The simple shared-coordinate hypothesis is not supported: corresponding action and video weights are not aligned by a direct crop. I will record that negative result and leave the checkpoint untouched while waiting for the simulator, rather than turn a weight-magnitude observation into an unsupported repair.
```python
with (W/"notes.md").open("a") as f:
    f.write("\n## Additional static negative results\n"
            "- Action weights are not simple top-left crops of video weights "
            "(tested text embeddings and attention projections; cosine near zero).\n"
            "- Removing video output coordinate 1938 algebraically reduces most "
            "late exceptional head OV singular values to ordinary ranges. "
            "This confirms the source of the spectral outliers, not their harmfulness.\n"
            "- No behavioral evidence is available yet; forward tool remains offline. "
            "All 40 passes remain. No edited weights are active.\n")
assert len(edited()) == 0
print("Analysis artifacts:",len(list(W.iterdir())))
print("Notes bytes:",(W/"notes.md").stat().st_size)
print("Active edits:",edited())
time.sleep(840)
print("Backend wait completed; ready to retry.")
```
