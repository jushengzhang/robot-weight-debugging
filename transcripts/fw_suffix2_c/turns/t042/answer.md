The earlier notes confirm that broad head removal and simple linear erasure did not restore clean behavior. I’ll read the remaining results and locate the saved probe-analysis code so the next comparison uses the same inputs and metrics.
```python
txt = (W / "prior_notes.md").read_text()
print("=== PRIOR NOTES, REMAINDER ===")
print(txt[txt.find("### Pass 23"):][-10500:])
print("=== WORK FILES ===")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    elif not p.name.startswith(("roll_", "probe_", "prior_candidate", "edited")):
        print(p.name + "/", "entries", len(list(p.iterdir())))
print("=== EDIT COUNT / FIRST NAMES ===")
ee = list(edited())
print(len(ee), ee[:8])
print("=== TEXT / PROPRIO TENSORS ===")
for n in NAMES:
    if "text_embedding" in n or "proprio" in n:
        print(n, SHAPES[n])
```
