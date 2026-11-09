Across the first 256 reserved tagged episodes, the edit is slightly behind the original, so there is no validated gain so far. I’ll complete the fixed validation plan; I won’t reinterpret the development result as evidence of success.
```python
s41=paired_stats([41]); pooled=paired_stats([40,41])
print("B",json.dumps(s41))
print("POOLED",json.dumps(pooled))
with (W/"notes.md").open("a") as f:
    f.write("\nFrozen validation raw B/pass41: "+json.dumps(s41)+". Pooled A+B: "+json.dumps(pooled)+". No weight changes; no validated gain so far.\n")
print(json.dumps(json.loads((W/"request42.json").read_text()),separators=(",",":")))
```
