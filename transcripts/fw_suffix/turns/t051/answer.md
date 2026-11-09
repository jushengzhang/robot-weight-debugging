I will save the sixth slice and move to the penultimate one.
```python
print(collect_context_slice(18,2560))
set_context_slice(3072)
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 18: context dimensions 2560:3072 acquired\n"
            "Temporary readout advances to 3072:3584.\n")
print("Next readout:", 3072, "to", 3584)
```
