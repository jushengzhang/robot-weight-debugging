The two safety batches total 237 original successes versus 234 edited successes, so they provide no reason to change the revert decision. I will save those results and locate the earlier probe inputs for one final check of whether the positional edit changes the suspected attention hotspot.

```python
safety_pairs = paired_records([38, 39])
assert {(o["id"], o["seed"]) for o, e in safety_pairs} == {
    (e["id"], e["seed"]) for b in safety_batches.values() for e in b
}
d_safety = np.array([
    int(bool(e["success"])) - int(bool(o["success"])) for o, e in safety_pairs
], dtype=float)
safety_diff = float(100 * d_safety.mean())
safety_se = float(100 * d_safety.std(ddof=1) / np.sqrt(len(d_safety)))
safety_summary = {
    "overall": paired_stats(safety_pairs),
    "approx_95pct_ci_pp": [round(safety_diff - 1.96 * safety_se, 3),
                           round(safety_diff + 1.96 * safety_se, 3)],
    "observed_drop_within_2pp": bool(safety_diff >= -2),
    "by_ptype": {
        t: paired_stats([(o, e) for o, e in safety_pairs if o["ptype"] == t])
        for t in ("object", "semantic")
    }
}
keep_candidate = bool(confirmation_passed and safety_diff >= -2)
evaluation_summary = {
    "candidate_hash": "44d71c5e3051dd09cd7473497428e25aa5194cef76caf0a0cedf49d3910f0916",
    "confirmation": final_summary,
    "safety": safety_summary,
    "keep_candidate": keep_candidate,
    "original_task_evaluation_available": False
}
(work / "evaluation_summary.json").write_text(json.dumps(evaluation_summary, indent=2))
print("Safety summary:", json.dumps(safety_summary, sort_keys=True))
print("Predeclared decision:", "KEEP" if keep_candidate else "REVERT AFTER DIAGNOSTIC PROBE")

with (work / "notes.md").open("a") as f:
    f.write(
        "\n## Passes38-39: independent safety checks (256 episodes)\n"
        "Batch1:118->117/128, 3 wins/4 losses. Batch2:119->117/128, 3 wins/5 losses.\n"
        + json.dumps(safety_summary) +
        "\nObserved combined safety drop is within2pp, but uncertainty does not establish "
        "2pp non-inferiority. Confirmation-gain criterion failed; final decision is REVERT. "
        "Only the already planned diagnostic probe remains before restoration.\n"
    )

prior_probe_index_path = work / "probe_13" / "index.json"
if prior_probe_index_path.exists():
    prior_probe_index = json.loads(prior_probe_index_path.read_text())
    print("Earlier keep20% probe index preview:", json.dumps(prior_probe_index)[:3500])
else:
    print("Probe13 index unavailable; saved diagnostic inputs:", json.dumps(diag_manifest))
print("Current frozen hash:",
      hashlib.sha256(load(current_edits[0]).contiguous().numpy().tobytes()).hexdigest())
```
