I’ll merge the two parts of this slice, verify the query scores against the reference slice, and move to coordinates 1920–2399.```python
s11, q11 = decode_readout(11,1440)
s12, q12 = decode_readout(12,1440)
sl = np.concatenate([s11,s12],axis=0)
sc = np.concatenate([q11,q12],axis=0)
assert sl.shape[0] == 72
np.save(readout_dir/"sharp_slice_1440.npy",sl)
np.save(readout_dir/"sharp_scores_1440.npy",sc)
print("Merged score mismatch:", rms(sc-scores0))
set_readout_slice(1920)
with (W/"notes.md").open("a") as f:
    f.write("\nPasses11-12: slice1440:1920 acquired in two input batches (24raw,48clean/language), "
            "merged in canonical order. Next1920:2400. One extra acquisition pass used.\n")
```
