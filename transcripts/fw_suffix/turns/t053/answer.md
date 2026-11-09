I will save the penultimate slice and configure the final 512 coordinates. After that, I will examine how much of the tagged/clean contrast lies outside the clean-language and proprioception spans.
```python
print(collect_context_slice(19,3072))
set_context_slice(3584)
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 19: context dimensions 3072:3584 acquired\n"
            "Temporary readout advances to final slice 3584:4096. "
            "After pass 20, remove ALL 16 instrumentation edits before applying a repair.\n")
print("Final readout:", 3584, "to", 4096)
```
