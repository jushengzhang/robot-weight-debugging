The complete clean check is 343 versus 346 successes out of 512. With tagged success at 229 versus 230, the large tag penalty remains essentially unchanged. I’ll save those results and load the first reserved Language Instructions batch.
```python
clean_final = record_validation(47, "strip_D", [44, 45, 46, 47], "clean")
results = json.loads((root / "validation_results.json").read_text())
results["tag_penalty_pp"] = {
    model: 100 * (clean_final[model] - raw_final[model]) / raw_final["n"]
    for model in ("original", "edited")
}
results["clean_point_loss_within_2pp"] = clean_final["diff_pp"] >= -2.0
(root / "validation_results.json").write_text(json.dumps(results, indent=2))
print("TAG_PENALTY_PP", results["tag_penalty_pp"])
with (root / "notes.md").open("a") as f:
    f.write("\nFinal clean A-D complete. Tag penalty (strip minus raw) pp: " +
            json.dumps(results["tag_penalty_pp"]) +
            ". No closed-loop tag-gap closure demonstrated. Remaining "
            "passes48-50 are untouched Language A-C; candidate remains frozen.\n")
show_next_request(48)
```
