Batch 4 lost three successes, leaving the first 512 episodes almost exactly tied. I will record the pooled result and load the last two final-practice batches, with no changes to the weights.

```python
pairs_so_far = paired_records([32, 33, 34, 35])
summary4 = {
    "overall": paired_stats(pairs_so_far),
    "by_ptype": {
        t: paired_stats([(o, e) for o, e in pairs_so_far if o["ptype"] == t])
        for t in sorted({o["ptype"] for o, e in pairs_so_far})
    }
}
print("Final batches 1-4:", json.dumps(summary4, sort_keys=True))
assert hashlib.sha256(load(current_edits[0]).contiguous().numpy().tobytes()).hexdigest() == \
    "44d71c5e3051dd09cd7473497428e25aa5194cef76caf0a0cedf49d3910f0916"

with (work / "notes.md").open("a") as f:
    f.write(
        "\n## Pass35: frozen edit, final batch4\n"
        "69->66/128; -2.3 +/-2.8pp paired SE; 5 wins/8 losses. "
        "Env14->12, object19->21, position7->6, semantic23 unchanged, task6->4. "
        "Pooled batches1-4: " + json.dumps(summary4["overall"]) +
        ". Frozen hash reverified; no changes.\n"
    )
print("Final batch5:", json.dumps(final_batches[5], separators=(",", ":")))
print("Final batch6:", json.dumps(final_batches[6], separators=(",", ":")))
```
