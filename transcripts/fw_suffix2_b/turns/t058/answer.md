Both experts contribute: action-only and video-only repairs each reduce the diagnostic gap, while repairing both gives a larger effect. I will now test the two-expert rank-256 candidate on the fixed 96-episode development set, which was not used to construct the projection, before pursuing further refinements.
```python
apply_context_candidate(256,1.0)
dev_eps=[{k:v[k] for k in ("suite","id","seed")} for v in sp["development"]]
print("DEVELOPMENT EPISODES")
print(json.dumps(dev_eps,separators=(",",":")))
practice=json.loads((Path(BENCH)/"practice_variants.json").read_text())
print("PRACTICE STRUCTURE",type(practice).__name__,
      list(practice)[:5] if isinstance(practice,dict) else practice[:1])
with (W/"notes.md").open("a") as f:
    f.write("\nCausal route localization rank256: action-only executed gap ratio.897 (clean/language drifts.00609/.00608); video-only.878 (drifts.00477/.02991); both.763. Camera large improvement only with both, showing nonlinear joint dependence. Video edit causes diagnostic paraphrase outlier. Pass22 planned fixed development96 paired raw, both-expert rank256 dose1, no gate. Development variants unused in context construction; reserved validation remains untouched.\n")
```
