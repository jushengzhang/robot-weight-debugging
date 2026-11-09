The double dose lost the standard dose’s gain: 47 versus 47 successes, with five gains and five regressions. Stronger suppression is therefore not automatically better. I’ll inspect detector coverage next and test whether a lower detection threshold can catch missed tag features without increasing the attention penalty.
```python
s25 = paired_rollout(25)
with (W / "notes.md").open("a") as f:
    f.write("\n### Pass25 video gate double-dose\n"
            + json.dumps(s25) +
            "\nDouble-dose rejected; standard dose remains development leader, not validated.\n")
print("Pass25:", s25)
for filename in ["dense_gate_geometry.pt", "dense_context_geometry.pt", "dense_gate_metadata.pt"]:
    data = torch.load(W / filename, weights_only=False)
    print(filename, small_summary(data))
print("\nANALYSIS HELPER:")
print((W / "analysis_helpers.py").read_text()[:4000])
print("\nREPLAY META:")
rmeta = json.loads((W / "replay_meta.json").read_text())
print(str(rmeta)[:1000])
print("\nPROBE ITEMS:")
pitems = json.loads((W / "probe_items.json").read_text())
print(str(pitems)[:800])
```
