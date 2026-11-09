The prior work found that linear projections barely changed the tag-induced action error, and it eventually moved to a nonlinear attention gate. I will read the gate's construction and validation results in full before deciding whether to extend it or test a different mechanism.
```python
prior = (W / "prior_notes.md").read_text()
print("NOTES LENGTH:", len(prior))
print(prior[9500:18500])
print("\nTENSOR COUNT:", len(NAMES), "CURRENT EDITS:", len(edited()))
print("\nNON-BLOCK TENSORS:")
for n in NAMES:
    if ".blocks." not in n:
        print(n, tuple(SHAPES[n]))
print("\nBLOCK 0 TENSORS:")
for n in NAMES:
    if ".blocks.0." in n:
        print(n, tuple(SHAPES[n]))
```
