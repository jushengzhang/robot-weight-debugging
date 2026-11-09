The saved traces already show a useful distinction: the robot-start pair has identical first inputs and actions until action 85, while the lighting pair starts with 22 different image elements. I will finish saving the safety audit, account for repeated sampled variants, and locate the probe metadata.
```python
# Finish the audit that was interrupted by an incorrect metadata filename.
latest_safety = {}
for p in safety_pairs:
    latest_safety[(p["suite"], p["id"], p["seed"])] = p
deduplicated_safety = list(latest_safety.values())
deduplicated_summary = {
    category: paired_summary([p for p in deduplicated_safety if p["category"] == category])
    for category in safety_categories}
print("DEDUPLICATED SAFETY:", json.dumps(deduplicated_summary, indent=2))
print("TOTAL PAIRED TRIALS / UNIQUE VARIANT-SEED PAIRS:",
      len(safety_pairs), len(deduplicated_safety))

(W / "F_final_safety_validation.json").write_text(json.dumps({
    "summary_all_trials": safety_summary,
    "summary_deduplicated_latest": deduplicated_summary,
    "episodes": safety_pairs,
    "unique_prompts": len(all_safety_texts), "edited_token_overlaps": 0}, indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\n## Final rollout safety totals (passes24-29 and43-48)\n"
            + json.dumps(safety_summary) +
            "\nUnique variant-seed pairs: " + str(len(deduplicated_safety)) +
            "; deduplicated latest-run summary: " + json.dumps(deduplicated_summary) +
            "\nAll50 safety prompts avoid edited rows. Robot-start447: identical initial "
            "policy inputs and first85 actions. Lighting2341: initial base images differ "
            "in22 elements, explaining why the two initial chunks need not match. "
            "Both negative discordances retained; exact-input probes next.\n")

print("PROBE30 METADATA FILES:", [p.name for p in (W / "probe_30").glob("*.json")])
for p in (W / "probe_30").glob("*.json"):
    meta30 = json.loads(p.read_text())
    print(p.name, str(meta30)[:2700])
print("PROBE30 ARRAY NAMES:", [p.name for p in sorted((W / "probe_30").glob("*.npy"))[:6]])
print("PASS35 DISCORDANCES:")
r35_lookup = {(r["suite"], r["id"], r["model"]): r for r in read_roll(35)}
for p in paired_roll(35):
    if p["original"] != p["edited"]:
        rr = r35_lookup[(p["suite"], p["id"], "original")]
        print(json.dumps({**p, "instruction": rr["instruction"]}))
```
