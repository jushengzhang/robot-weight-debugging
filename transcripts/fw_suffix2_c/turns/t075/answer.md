Inspection shows that the saved detector used threshold 1.0, not 0.5 as the notes suggested. I’ll explicitly use 0.5 for this diagnostic and add an unmasked readout that measures detector activity, so we can verify its response on clean and paraphrased inputs rather than assume it is selective.
```python
masked_spec["detector_bias"]=-16.0  # gain32 times threshold0.5
masked_spec["queries"]=query_spec+[(-2,0.0)]  # final readout is uniform, unmasked
torch.save(masked_spec,W/"masked_readout_spec.pt")
needle='''        a[anchors,m+3]=-128.
        if direction>=0:
            a[anchors,m+direction]=gain'''
replacement='''        a[anchors,m+3]=-128.
        if direction>=0:
            a[anchors,m+direction]=gain
        if direction==-2:
            a[anchors,m+3]=0
            a[-1]=load_original(cp+"k.weight")[-1]'''
assert needle in masked_code
masked_code=masked_code.replace(needle,replacement)
(W/"masked_readout.py").write_text(masked_code)
exec(masked_code)
install_masked_readout(0)
with (W/"notes.md").open("a") as f:
    f.write("\nCorrection: saved original readout detector was threshold1.0 (norm32,bias-32). New masked readout explicitly uses threshold.5,bias-16. Added block13 uniform unmasked readout to measure detector activity on clean/paraphrases. Request layers0..13.\n")
```
