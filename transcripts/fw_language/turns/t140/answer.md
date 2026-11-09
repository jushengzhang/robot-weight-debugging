The completed language test gained only one net success across 494 paired episodes (+0.2 points), while Robot Initial States and Sensor Noise regressed by 3.1 and 2.7 points. The candidate fails the locked retention rule. I need one cleanup operation to remove it and verify that the submitted weights are the originals; no further experiments or tuning are involved.
```python
# Final selection only: summarize the completed evaluation and restore originals.
verify_frozen_candidate()
rows40 = read_roll(40)
for model in ("original", "edited"):
    assert episode_keys([r for r in rows40 if r["model"] == model]) == episode_keys(replication_batches[1])

ledger["language"]["19"] = paired_summary([39, 40])
combined = paired_summary([25, 26, 39, 40])

outcomes = {}
for k in [25, 26, 39, 40]:
    for r in read_roll(k):
        key = (r["suite"], r["id"], r["seed"])
        assert r["model"] not in outcomes.setdefault(key, {})
        outcomes[key][r["model"]] = int(bool(r["success"]))

by_variant = {}
for (suite, variant_id, seed), values in outcomes.items():
    assert set(values) == {"original", "edited"}
    by_variant.setdefault((suite, variant_id), {})[seed] = values["edited"] - values["original"]
assert len(by_variant) == 247
assert all(set(v) == {7, 19} for v in by_variant.values())

cluster_means = np.array([np.mean(list(v.values())) for v in by_variant.values()])
cluster_se = float(100 * cluster_means.std(ddof=1) / np.sqrt(len(cluster_means)))
combined["unique_variants"] = len(by_variant)
combined["seeds"] = [7, 19]
combined["variant_clustered_se_pp"] = round(cluster_se, 4)
combined["variant_clustered_approx_95ci_pp"] = [
    round(float(100 * cluster_means.mean() - 1.96 * cluster_se), 4),
    round(float(100 * cluster_means.mean() + 1.96 * cluster_se), 4),
]
ledger["language"]["combined"] = combined

language_passes_rule = float(100 * cluster_means.mean()) > 2 * cluster_se
safety_losses = {c: s["diff_pp"] for c, s in ledger["safety"].items() if s["diff_pp"] < -2}
assert not language_passes_rule
assert safety_losses

# Keep the rejected candidate backup outside the submitted edit directory.
assert (root / "candidate_action_L11H5_cross_O_half.pt").is_file()
removed_names = list(edited())
for tensor_name in removed_names:
    drop_edit(tensor_name)
assert len(edited()) == 0
for tensor_name in removed_names:
    assert torch.equal(load(tensor_name), load_original(tensor_name))

decision = {
    "selected": "original",
    "active_edits": list(edited()),
    "candidate_rejected": ledger["candidate"],
    "reason": "No independent language gain above uncertainty; observed non-language regressions exceed 2 pp.",
    "language_gain_above_two_clustered_se": language_passes_rule,
    "safety_losses_over_two_pp": safety_losses,
    "language_confirmation": combined,
    "forward_passes_used": 40,
    "restored_tensors_verified_equal_to_original": removed_names,
}
ledger["final_decision"] = decision
ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
(root / "final_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
(root / "active_candidate.json").write_text(json.dumps({
    "label": "original_no_edits",
    "active_edits": [],
    "reason": decision["reason"]
}, indent=2) + "\n")

with (root / "notes.md").open("a") as f:
    f.write("\n## Pass 40 and FINAL SELECTION\n"
            + "- Pass 40: " + json.dumps(paired_summary([40])) + "\n"
            + "- Language seed19: " + json.dumps(ledger["language"]["19"]) + "\n"
            + "- Complete independent language confirmation: " + json.dumps(combined) + "\n"
            + "- Candidate REJECTED under the locked rule. All edits removed; restored tensor verified exactly equal to original.\n"
            + "- FINAL SUBMISSION: ORIGINAL WEIGHTS, zero active edits. No demonstrated robustness improvement.\n"
            + "- Evidence and final decision saved in locked_evaluation_results.json and final_decision.json.\n")

print("Language seed19:", json.dumps(ledger["language"]["19"]))
print("Combined language:", json.dumps(combined))
print("Safety losses beyond 2 pp:", json.dumps(safety_losses))
print("FINAL SUBMISSION: original weights; active edits =", edited())
```
