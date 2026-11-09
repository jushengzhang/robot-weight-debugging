Batch 3 gained three successes, but the pooled improvement is still small relative to its uncertainty. I will log the result and prepare batch 4. I will also reserve pass 40 for an exact-input attention probe of the frozen edit, to check its internal effect without testing another candidate.

```python
pairs_so_far = paired_records([32, 33, 34])
summary3 = {
    "overall": paired_stats(pairs_so_far),
    "by_ptype": {
        t: paired_stats([(o, e) for o, e in pairs_so_far if o["ptype"] == t])
        for t in sorted({o["ptype"] for o, e in pairs_so_far})
    }
}
print("Final batches 1-3:", json.dumps(summary3, sort_keys=True))
diag_manifest = json.loads((work / "diag_inputs.json").read_text())
print("Saved diagnostic manifest:", json.dumps(diag_manifest)[:1800])

with (work / "notes.md").open("a") as f:
    f.write(
        "\n## Pass34: frozen edit, final batch3\n"
        "81->84/128; +2.3 +/-2.6pp paired SE; 7 wins/4 losses. "
        "Env17->20, object20->21, position9->6, semantic24->25, task11->12. "
        "Pooled batches1-3: " + json.dumps(summary3["overall"]) + ".\n"
        "\n## Remaining-pass allocation\n"
        "Passes35-37: final batches4-6; passes38-39: preselected safety batches1-2. "
        "Pass40: exact saved-input attention probe of the same frozen five-row keep50% edit "
        "(no tuning; diagnostic only, excluded from confirmation statistics), then apply "
        "the predeclared keep/revert rule. No new candidate will be introduced.\n"
    )
print("Final batch4:", json.dumps(final_batches[4], separators=(",", ":")))
```
