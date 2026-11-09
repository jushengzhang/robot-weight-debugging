I’ll archive the third slice and advance to coordinates 2688–3583. After the fifth slice, I’ll remove every scanner edit before testing a repair.
```python
_,y=dense_extract(15,1792)
print("Slice1792 finite:",np.isfinite(y).all(),"RMS:",round(float(np.sqrt((y*y).mean())),5))
advance_dense_slice(2688)
with (W/"notes.md").open("a") as f:
    f.write("\nPasses14-15: dense slices896:1792 and1792:2688 saved; finite readouts. Next2688:3584. All522 current edits are instrumentation, not a repair.\n")
print("Next slice:2688:3584")
```
