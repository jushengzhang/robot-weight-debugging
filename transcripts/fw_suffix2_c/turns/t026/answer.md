I will save the middle slice and switch to the final slice. After the next probe I will remove the entire diagnostic circuit, then use only the measured statistics to construct a targeted candidate.
```python
f9=extract_features(9,1)
instrument_features(2)
with (W/"notes.md").open("a") as f:
    f.write("\nPass8-9 readouts validated: temporal RMS below1e-6; masking leaves measured clean/paraphrase means exactly unchanged. "
            "Slice0 uniform tag contrast RMS.00633, masked.00439: conservative suffix masking leaves substantial context contamination. "
            "Final slice pending; instrumentation is still active and must be cleared.\n")
```
