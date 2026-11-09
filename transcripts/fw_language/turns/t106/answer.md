Complete removal performs worse, so the mode appears to carry useful information as well as wording sensitivity. I will restore the conservative half-retention edit and freeze it. The remaining passes will evaluate this one candidate: 247 untouched language variants at two seeds, plus paired checks of all six other categories.
```python
s24 = paired_summary(paired_records([24]))
print("Zero-retention discovery:", s24)
print("Locked candidate:", set_text_mode_retention(0.5))

confirmation_episodes = frozen_batches[3] + frozen_batches[4]
seen_language = set()
for path in work.glob("roll_*.jsonl"):
    for r in read_jsonl(path):
        if r.get("category") == "Language Instructions":
            seen_language.add((r["suite"], int(r["id"])))
assert confirmation_keys.isdisjoint(seen_language)
assert len(confirmation_keys) == 247

final_plan = {
    "candidate": json.loads((work / "active_candidate.json").read_text()),
    "no_further_tuning": True,
    "language": [
        {"stage": "untouched_variants_seed7", "batches": frozen_batches[3:]},
        {"stage": "seed_replication19", "batches":
         [[dict(e, seed=19) for e in b] for b in frozen_batches[3:]]}
    ],
    "safety_plan": "256 unique practice variants per non-language category, paired; "
                   "two disjoint batches of 128 each",
    "retention_rule": "Independent language gain > 2 paired SE, with replication "
                      "clustered by variant; no material non-language harm. "
                      "Otherwise restore original. Do not tune on these results."
}
(work / "final_locked_evaluation_plan.json").write_text(json.dumps(final_plan, indent=2))
with (work / "notes.md").open("a") as f:
    f.write("\n## Pass 24 and FINAL CANDIDATE LOCK\n"
            "- Full removal of action L11 H5 leading O mode: " + json.dumps(s24) + "\n"
            "- Restore half retention for evaluation, not acceptance. "
            "Fixed-input mode contrast was reduced to .5044 (first step), "
            "but discovery success gain +2.3 pp was below SE.\n"
            "- No further weight tuning. Remaining 16 passes: two batches of "
            "untouched language variants (128+119) at seed7, twelve safety batches "
            "(256 unique variants per other category), and the same untouched "
            "language variants at seed19. Final decision uses all predeclared results.\n"
            "- If independent gain is not clearly larger than paired uncertainty "
            "or there is material non-language damage, submit original.\n")
print("First untouched confirmation batch:")
print(json.dumps(frozen_batches[3], separators=(",", ":")))
```
