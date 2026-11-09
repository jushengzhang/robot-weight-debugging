The prior session found that a thresholded feature could distinguish suffix-bearing tokens from clean text, but attention gating alone gave only a small gain. I’ll recover the current readout scripts and finish checking the masked features before deciding whether the newer projection approach is worth validating.
```python
s = (W / "prior_notes.md").read_text()
print(s[10000:16500])
print("\nCURRENT EDIT COUNT:", len(edited()))
print("\nRELEVANT FILES")
for p in sorted(W.iterdir()):
    if p.name.endswith((".py", ".pt", ".json")) or p.name.startswith("candidate"):
        print(p.name, p.stat().st_size)
print("\nPROBE 28 STRUCTURE")
r28 = json.loads((W / "probe_28/results.json").read_text())
print(str(r28)[:1600])
print("\nMASKED READOUT SCRIPT")
print((W / "masked_readout.py").read_text()[:11000])
```
