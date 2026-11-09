The earlier session found that broad text ablation and mean-context projections changed clean behavior without reliably correcting tagged behavior. I will recover the newer, richer context analysis and its probe setup so the next test isolates the current two-tensor candidate.
```python
prior = (W / "prior_notes.md").read_text()
print("PRIOR NOTES, REMAINING", prior[prior.find("### pass 23"):] if "### pass 23" in prior else prior[-5100:])
print("\nFILES")
for p in sorted(W.iterdir()):
    print(p.name + ("/" if p.is_dir() else ""), len(list(p.iterdir())) if p.is_dir() else p.stat().st_size)
print("\nACTIVE EDITS", list(edited()))
print("\nTEXT/PROPRIO TENSORS")
for n in NAMES:
    if "text_embedding" in n or "proprio" in n:
        print(n, SHAPES[n])
print("\nSAVED SCRIPTS")
for p in W.glob("*.py"):
    print(p.name)
print("\nSPLITS")
sp = json.loads((W/"splits.json").read_text())
print({k: len(v) if hasattr(v, "__len__") else v for k,v in sp.items()})
```
