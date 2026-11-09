All three language-evaluation seeds are complete. I will calculate the final paired and variant-clustered results, including hard-instruction subsets, then prespecify the additional safety sweep. No weight changes will be made from these results.
```python
seed6203_pairs = sum([audit_prespecified_pass(k) for k in range(39, 43)], [])
assert len(seed6203_pairs) == 460
all_validation = seed7_pairs + seed3101_pairs + seed6203_pairs
assert len(all_validation) == 1380

texts = {}
for k in range(35, 43):
    for r in read_roll(k):
        if r["model"] == "original":
            key = (r["suite"], r["id"])
            if key in texts:
                assert texts[key] == r["instruction"]
            texts[key] = r["instruction"]
for p in all_validation:
    text = texts[(p["suite"], p["id"])]
    ids = tok.encode(text.strip().lower().replace("_", " ") + "\n", add_bos=True)
    p["edited_token_present"] = bool(set(ids) & set(changed))

final_language_summary = {
    "overall": cluster_summary(all_validation),
    "by_seed": {str(seed): paired_summary([p for p in all_validation if p["seed"] == seed])
                for seed in [7, 3101, 6203]},
    "by_suite": {suite: cluster_summary([p for p in all_validation if p["suite"] == suite])
                 for suite in sorted({p["suite"] for p in all_validation})},
    "difficulty_4_5": cluster_summary([p for p in all_validation if int(p["difficulty"]) >= 4]),
    "difficulty_5": cluster_summary([p for p in all_validation if int(p["difficulty"]) == 5]),
    "by_token_coverage": {
        str(touched): cluster_summary([p for p in all_validation if p["edited_token_present"] == touched])
        for touched in [False, True]}
}

# Nonparametric uncertainty check; analysis only, no weight fitting.
cluster_values = collections.defaultdict(list)
for p in all_validation:
    cluster_values[(p["suite"], p["id"])].append(int(p["edited"]) - int(p["original"]))
dbar = np.array([np.mean(x) for x in cluster_values.values()])
rng = np.random.default_rng(202601)
boot = np.mean(rng.choice(dbar, size=(20000, len(dbar)), replace=True), axis=1) * 100
final_language_summary["overall"]["variant_bootstrap_95ci_pp"] = [
    round(float(x), 3) for x in np.quantile(boot, [0.025, 0.975])]
print(json.dumps(final_language_summary, indent=2))
(W / "F_final_language_validation.json").write_text(json.dumps({
    "summary": final_language_summary, "episodes": all_validation}, indent=2))

safety_categories = ["Camera Viewpoints", "Robot Initial States", "Light Conditions",
                     "Background Textures", "Objects Layout", "Sensor Noise"]
safety_plan = {}
for k, category in enumerate(safety_categories, start=43):
    req = {"type": "rollout", "model": "both",
           "sample": {"category": category, "n": 128, "seed": 9107},
           "save_frames": True}
    safety_plan[str(k)] = req
(W / "F_safety_replication_plan.json").write_text(json.dumps(safety_plan, indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\n## Three-seed independent language validation completed (pass42)\n"
            + json.dumps(final_language_summary["overall"]) +
            "\nBy suite: " + json.dumps(final_language_summary["by_suite"]) +
            "\nHard levels4-5: " + json.dumps(final_language_summary["difficulty_4_5"]) +
            "\nF has remained unchanged throughout. Next six paired safety batches: "
            "128/category, sample seed9107, save genuine inputs for fixed-input probes.\n")
```
