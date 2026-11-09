I will save coordinates 1536–2047 and advance to 2048–2559. The intervention will be computed only after the full context contrast is available, so its direction will not depend on an arbitrary subset of embedding coordinates.
```python
print(collect_context_slice(16,1536))
set_context_slice(2048)
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 16: context dimensions 1536:2048 acquired\n"
            "Temporary readout advances to 2048:2560. Probe layers must be integer indices.\n")
print("Next readout:", 2048, "to", 2560)
```
