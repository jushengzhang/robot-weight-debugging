I’ll save the second slice and install the final one. After that, I’ll remove all readout edits and construct the pre-GELU candidate from the recovered summaries.
```python
neg1=collect_negative_readout(32)
install_negative_readout(2)
with (W/"notes.md").open("a") as f:
    f.write("\nPass32 negative video slice1 saved, temporally stable. Temporary instrumentation advanced to final slice2 for pass33.\n")
```
