I’ll compare the new gate with the earlier single-detector gate. If stronger suppression worsens the action match, I’ll narrow the gate rather than assuming that suppressing more text is better.
```python
m32=probe_metrics(32)
print(json.dumps(m32,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass32 four-detector gate strength12\n"+json.dumps(m32)+"\n")
print("Remaining reserved episode counts:")
for fn in ("development_episodes.json","validation_a_episodes.json","validation_b_episodes.json",
           "language_a_episodes.json","language_b_episodes.json"):
    d=json.loads((W/fn).read_text())
    print(fn,len(d),str(d[:2]) if isinstance(d,list) else list(d))
```
