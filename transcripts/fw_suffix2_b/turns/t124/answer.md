My aggregation grouped the paraphrase batches with tagged batches because both use the `raw` language setting. I will separate them by their preplanned split names. The clean-instruction result is unaffected: 528 to 508 successes out of 768, a 2.60-point loss.
```python
validation_pairs = {}
validation_passes = {}
for entry in validation_plan:
    pid = entry["pass"]
    if not (root / ("roll_%d.jsonl" % pid)).exists():
        continue
    condition = "language" if entry["split"].startswith("language_") else entry["condition"]
    pairs = read_pairs(pid)
    assert len(pairs) == 128
    validation_pairs.setdefault(condition, []).extend(pairs)
    validation_passes.setdefault(condition, []).append(pid)

validation_totals = {}
for condition, pairs in validation_pairs.items():
    assert len({p[0] for p in pairs}) == len(pairs)
    expected_language = condition == "language"
    assert all((p[1]["category"] == "Language Instructions") == expected_language for p in pairs)
    validation_totals[condition] = summarize_pairs(pairs)
    validation_totals[condition]["passes"] = validation_passes[condition]

print("CORRECTED TOTALS THROUGH PASS 49:")
print(json.dumps(validation_totals, ensure_ascii=True, indent=1))

print("\nCURRENT SESSION NOTES, beginning:")
print(notes_text[:6000])

frozen_manifest = json.loads((root / "frozen_manifest.json").read_text())
print("\nMANIFEST TYPE:", type(frozen_manifest).__name__)
if isinstance(frozen_manifest, dict):
    print("MANIFEST LENGTH:", len(frozen_manifest))
    print("MANIFEST FIRST TWO ENTRIES:", repr(list(frozen_manifest.items())[:2])[:500])
print("FINAL PLAN SPLIT:", last_plan_entry["split"])
```
