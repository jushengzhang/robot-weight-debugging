Batch 1 showed a loss of 3.1 percentage points, with a paired standard error of 2.5 points. I will log it and check the saved batch lists before continuing; the edit remains frozen, and the decision will use all six final batches rather than this result alone.

```python
final_batches = {
    i: json.loads((work / f"final_batch_{i}.json").read_text())
    for i in range(1, 7)
}
safety_batches = {
    i: json.loads((work / f"safety_batch_{i}.json").read_text())
    for i in range(1, 3)
}
all_planned = [
    (e["id"], e["seed"])
    for batch in list(final_batches.values()) + list(safety_batches.values())
    for e in batch
]
assert all(len(b) == 128 for b in final_batches.values())
assert all(len(b) == 128 for b in safety_batches.values())
assert len(set(all_planned)) == 1024
assert len({e[0] for e in all_planned}) == 1024
assert hashlib.sha256(load(current_edits[0]).contiguous().numpy().tobytes()).hexdigest() == \
    "44d71c5e3051dd09cd7473497428e25aa5194cef76caf0a0cedf49d3910f0916"
print("Plan audit: 768 final + 256 safety episodes; all IDs distinct; frozen hash intact.")

records32 = [json.loads(line) for line in (work / "roll_32.jsonl").read_text().splitlines() if line.strip()]
print("Rollout rows:", len(records32))
print("Record keys:", list(records32[0]))
print("Small fields:", {
    k: v for k, v in records32[0].items()
    if k in ("id", "episode_id", "seed", "model", "ptype", "success", "suite", "task")
})
print("Nested field keys:", {
    k: list(v)[:20] for k, v in records32[0].items() if isinstance(v, dict)
})

with (work / "notes.md").open("a") as f:
    f.write(
        "\n## Pass32: frozen edit, final batch1\n"
        "77->73/128; -3.1 +/-2.5pp paired SE; 3 wins/7 losses. "
        "Env11->10, object22->23, position8->7, semantic24->21, task12 unchanged. "
        "Old-goal failures9->10. Frozen edit unchanged; all six batches will be pooled "
        "before the predeclared keep/revert decision. Planned 1024 IDs are distinct.\n"
    )
print("Final batch2:", json.dumps(final_batches[2], separators=(",", ":")))
```
