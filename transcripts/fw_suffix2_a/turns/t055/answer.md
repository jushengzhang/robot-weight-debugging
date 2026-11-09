I’ll save the second slice and advance to coordinates 1792–2687. The selectors remain unchanged; only the measured coordinates move.
```python
_,y=dense_extract(14,896)
print("Slice896 finite:",np.isfinite(y).all(),"RMS:",round(float(np.sqrt((y*y).mean())),5))
advance_dense_slice(1792)
print("Next slice:1792:2688; diagnostic edits:",len(edited()))
```
