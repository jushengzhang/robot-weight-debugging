I will save this slice and advance the readout. Once acquisition is complete, all diagnostic edits will be cleared before I apply a low-rank change at the two text-entry projections.
```python
print(collect_context_slice(17,2048))
set_context_slice(2560)
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 17: context dimensions 2048:2560 acquired\n"
            "Temporary readout advances to 2560:3072.\n")
print("Next readout:", 2560, "to", 3072)
```
