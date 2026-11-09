I will record the intervention and validation plan explicitly, then wait for a longer startup interval. In particular, a candidate must beat an equal-sized control-head edit and show a dose response before I spend the reserved passes on disjoint validation.

```python
plan = {
    "diagnostic_rollout": "102 predeclared episodes; original weights; residuals and exact saved inputs",
    "fixed_input_probes": "Matched first observations with identical diffusion noise; residual and attention outputs",
    "head_localization": "Attention maps at layers selected from fixed-input contrasts; compare texture-changed image regions",
    "candidate_families": [
        "Head-local attenuation of action self-attention output columns",
        "Low-rank nuisance-direction removal restricted to implicated head output blocks"
    ],
    "causal_tests": [
        "At least two nonzero doses plus original",
        "Matched control-head intervention",
        "Probe: background action deviation should shrink more than control action changes",
        "Rollout: same variants and seeds, paired success differences"
    ],
    "validation": {
        "selection_independent": True,
        "background": "At least 256 distinct reserved practice variants if feasible",
        "other_categories": "All six; paired comparisons, preferably 256 per category",
        "uncertainty": "Paired standard errors; variant-clustered errors for repeated seeds",
        "reject_rule": "Do not keep an edit with unclear background benefit or material collateral regressions"
    },
    "constraints": "No training, no Python model forward, no benchmark files except practice_variants.json"
}
(W/"experiment_plan.json").write_text(json.dumps(plan,indent=2))
note("## Locked experimental principles\n"
     "- Fixed-input contrasts precede causal edits.\n"
     "- Require original + two doses and a matched non-implicated-head control.\n"
     "- Reserve roughly half the 40-pass budget for selection-independent background and all-category validation.\n"
     "- No edit will be justified solely by large weight norms or singular values.\n"
     "- Backend still refused the pilot; no charged passes.")
print("Plan saved. Waiting 600 seconds for backend startup; weights remain original.")
time.sleep(600)
print("Wait complete; tensors edited:",len(edited()))
```
