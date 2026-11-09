The completed confirmation set has 434 successes for the original model and 433 for the edit. That fails the predeclared improvement requirement. I will finish the planned safety and attention checks for the record, then restore the original weights rather than submit an unsupported repair.

```python
from collections import Counter

final_pairs = paired_records(range(32, 38))
assert {(o["id"], o["seed"]) for o, e in final_pairs} == {
    (e["id"], e["seed"]) for b in final_batches.values() for e in b
}
d_final = np.array([
    int(bool(e["success"])) - int(bool(o["success"])) for o, e in final_pairs
], dtype=float)
final_diff = float(100 * d_final.mean())
final_se = float(100 * d_final.std(ddof=1) / np.sqrt(len(d_final)))
confirmation_passed = final_diff > 1.96 * final_se

final_summary = {
    "overall": paired_stats(final_pairs),
    "approx_95pct_ci_pp": [round(final_diff - 1.96 * final_se, 3),
                           round(final_diff + 1.96 * final_se, 3)],
    "confirmation_passed": bool(confirmation_passed),
    "by_ptype": {
        t: paired_stats([(o, e) for o, e in final_pairs if o["ptype"] == t])
        for t in sorted({o["ptype"] for o, e in final_pairs})
    },
    "by_suite": {
        s: paired_stats([(o, e) for o, e in final_pairs if o["suite"] == s])
        for s in sorted({o["suite"] for o, e in final_pairs})
    }
}
(work / "confirmation_summary.json").write_text(json.dumps(final_summary, indent=2))
print("Complete frozen-candidate confirmation:", json.dumps(final_summary, sort_keys=True))

practice_raw = json.loads((Path(BENCH) / "practice_episodes.json").read_text())
practice_records = practice_raw if isinstance(practice_raw, list) else practice_raw["episodes"]
meta_by_id = {int(e["id"]): e for e in practice_records}
assert all(i in meta_by_id for i, seed in all_planned)
print("Allowed practice types:", sorted({e["ptype"] for e in practice_records}))
for i, batch in safety_batches.items():
    counts = Counter(meta_by_id[e["id"]]["ptype"] for e in batch)
    assert counts == Counter({"object": 64, "semantic": 64}), counts
    print("Safety batch", i, "type counts:", dict(counts))

with (work / "notes.md").open("a") as f:
    f.write(
        "\n## Passes36-37: frozen edit, final batches5-6\n"
        "Batch5:68->70/128, +1.6 +/-2.7pp; 7 wins/5 losses. "
        "Batch6:73->71/128, -1.6 +/-2.5pp; 4 wins/6 losses.\n"
        "\n## Completed independent confirmation (768 episodes)\n"
        + json.dumps(final_summary) +
        "\nThe confirmation requirement FAILED. Under the frozen decision rule, restore "
        "original weights after finishing the already planned safety batches and diagnostic "
        "probe; safety results cannot rescue a failure of the overall-gain requirement. "
        "Candidate remains temporarily installed only for those remaining checks. "
        "Allowed practice list contains no original/unperturbed ptype.\n"
    )
print("Safety batch1:", json.dumps(safety_batches[1], separators=(",", ":")))
print("Safety batch2:", json.dumps(safety_batches[2], separators=(",", ":")))
```
