I will save the second stable context slice and move the readout to coordinates 1024–1535. The only purpose of these temporary edits is to collect the frozen encoder’s context statistics; they are not being evaluated as robot policies.
```python
collect_context_slice(14,512)
install_context_readout(1024)
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 14: context dimensions 512:1024 acquired\n"
            "Stored context_slice_0512.npy; temporary readout now 1024:1536.\n")
```
