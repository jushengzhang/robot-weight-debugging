The higher threshold gives the best action match so far: a 3.4% reduction in the tagged-to-clean gap, with roughly 0.0013 RMS clean/paraphrase drift. That is modest evidence, not a success-rate result. I’ll freeze this candidate and use the remaining seven passes for independent tagged, stripped, and paraphrase rollouts, with no further tuning.
```python
frozen_meta=json.loads((W/"current_edit.json").read_text())
frozen_meta["frozen_after_pass"]=33
frozen_meta["weight_snapshot"]="WORK/fwd/probe_33_weights"
frozen_meta["selection_data"]="diagnostic probes only; validation sets untouched"
frozen_meta["validation_plan"]={
    "tagged":"validation_a, validation_b, new disjoint validation_c (384 paired episodes)",
    "strip":"validation_a and validation_b (256 paired episodes)",
    "language":"language_a and language_b (256 paired episodes)"}
(W/"frozen_candidate.json").write_text(json.dumps(frozen_meta,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Candidate frozen after pass33\n"
            "Single detector threshold1.0,strength6,gain32,key64,both experts;308 tensors. "
            "Immutable copy in fwd/probe_33_weights. No further tuning. Planned remaining "
            "seven passes: raw validationA/B/C (384 paired), stripA/B (256 paired), "
            "languageA/B (256 paired). Retain only with independent positive success evidence "
            "and no material clean/paraphrase loss; otherwise restore original.\n")
spl=json.loads((W/"splits.json").read_text())
print("Split structure:",{k:(len(v),type(v).__name__) for k,v in spl.items()})
pv=json.loads((pathlib.Path(BENCH)/"practice_variants.json").read_text())
print("Practice structure:",type(pv).__name__,
      (list(pv)[:8] if isinstance(pv,dict) else str(pv[:1])[:650]))
print("VALIDATION_A_EPISODES:")
print(json.dumps(json.loads((W/"validation_a_episodes.json").read_text()),
                 separators=(",",":")))
```
