Batch 2 gained 2.3 points, offsetting most of batch 1's loss. I will compute the pooled result directly from paired records and verify that none of the planned final or safety episode IDs appeared in earlier rollouts.

```python
def read_rollout(k):
    return [
        json.loads(line)
        for line in (work / f"roll_{k}.jsonl").read_text().splitlines()
        if line.strip()
    ]

def paired_records(passes):
    grouped = {}
    for k in passes:
        for r in read_rollout(k):
            key = (int(r["id"]), int(r["seed"]))
            assert r["model"] not in grouped.setdefault(key, {}), (k, key, r["model"])
            grouped[key][r["model"]] = r
    assert all(set(g) == {"original", "edited"} for g in grouped.values())
    return [(g["original"], g["edited"]) for g in grouped.values()]

def paired_stats(pairs):
    n = len(pairs)
    original = sum(bool(o["success"]) for o, e in pairs)
    edited_count = sum(bool(e["success"]) for o, e in pairs)
    d = np.array([int(bool(e["success"])) - int(bool(o["success"])) for o, e in pairs])
    se = float(100 * d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": n, "original": original, "edited": edited_count,
        "diff_pp": round(float(100 * d.mean()), 3),
        "se_pp": round(se, 3),
        "wins": int((d == 1).sum()), "losses": int((d == -1).sum())
    }

seen_before_freeze = set()
for p in work.glob("roll_*.jsonl"):
    k = int(p.stem.split("_")[1])
    if k <= 31:
        seen_before_freeze.update(int(r["id"]) for r in read_rollout(k))
overlap = seen_before_freeze & {i for i, seed in all_planned}
assert not overlap, sorted(overlap)
print("Independence audit: no planned final/safety ID appeared in passes 1-31.")

pairs_so_far = paired_records([32, 33])
assert {(o["id"], o["seed"]) for o, e in pairs_so_far} == {
    (e["id"], e["seed"]) for b in (final_batches[1], final_batches[2]) for e in b
}
summary = {
    "overall": paired_stats(pairs_so_far),
    "by_ptype": {
        t: paired_stats([(o, e) for o, e in pairs_so_far if o["ptype"] == t])
        for t in sorted({o["ptype"] for o, e in pairs_so_far})
    }
}
print("Final batches 1-2:", json.dumps(summary, sort_keys=True))
with (work / "notes.md").open("a") as f:
    f.write(
        "\n## Pass33: frozen edit, final batch2\n"
        "66->69/128; +2.3 +/-2.1pp paired SE; 5 wins/2 losses. "
        "Env12->11, object22 unchanged, position4->7, semantic23 unchanged, task5->6. "
        "Pooled batches1-2: " + json.dumps(summary["overall"]) + ". "
        "Audit confirms no final/safety ID appeared in passes1-31; no edits changed.\n"
    )
print("Final batch3:", json.dumps(final_batches[3], separators=(",", ":")))
```
