Static analysis has reached its useful limit without activations. I will record a fixed causal-testing plan and pause for backend startup rather than make speculative edits. The plan compares identical observations under two instructions, tests a narrow intervention at several strengths, and reserves most passes for paired validation.
```python
plan = {
    "diagnosis": [
        "Original language rollouts with saved inputs and layer residuals",
        "Original less-perturbed controls covering the same 40 canonical tasks",
        "Fixed-observation canonical/paraphrase probes with identical diffusion noise",
        "Attention-map inspection at up to four localized layers"
    ],
    "causal_tests": [
        "Edit only a localized instruction-sensitive mode, not whole-tensor scaling",
        "Measure a zero/mild/medium/strong dose-response",
        "Use a nearby non-implicated mode or alternative route as a negative control",
        "Check preservation of canonical-instruction action predictions"
    ],
    "validation": [
        "Lock candidate before fresh paired language and category evaluations",
        "Report language effects on variants excluded from discovery",
        "Use paired same-variant same-seed evaluations on all six non-language categories",
        "Revert if language gain is not distinguishable from paired uncertainty or controls degrade"
    ],
    "budget": {"total":40, "target_max_discovery":16, "target_validation":24},
    "weights_changed": []
}
(W/"causal_test_plan.json").write_text(json.dumps(plan,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Backend wait\n"
            "Still offline after 10 weight-analysis code runs. No charged forward passes "
            "and no edits. Static analysis is complete enough to define causal tests; "
            "pausing briefly rather than drawing conclusions without behavior.\n")
print(json.dumps(plan["budget"]))
print("Current edits:",edited())
time.sleep(180)
print("Backend startup pause finished.")
```
