I spotted a path-copying error in the last request: eight tagged inputs were replaced by their clean counterparts. I’ll identify the affected rows and remeasure only those eight inputs before assembling the data.
```python
rr=json.loads((W/"probe_17"/"results.json").read_text())
print("Result structure:",type(rr).__name__, list(rr)[:8] if isinstance(rr,dict) else len(rr))
print("First record:",str(rr[0] if isinstance(rr,list) else next(iter(rr.values())))[:1000])
a17,y17=dense_extract(17,3584)
print("Missing correct tagged rows:",list(range(16,24)))
print("Current diagnostic tensors:",len(edited()))
with (W/"notes.md").open("a") as f:
    f.write("\n### Passes16-17 dense acquisition\nFourth/fifth slices saved. Pass17 request accidentally used clean paths at tagged rows16:24; "
            "must replace these eight final512-coordinate readouts with pass18 before assembly. Instrumentation still active.\n")
```
