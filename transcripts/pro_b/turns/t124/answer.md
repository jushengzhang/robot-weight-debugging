Batch 4 improved from 73 to 77 successes, bringing the cumulative result to 306 versus 309 across 512 episodes—a small, inconclusive gain. I’ll record that result and retrieve the final prespecified batch.
```python
def read_pairs(pass_num):
    rows = [
        json.loads(s)
        for s in (work / f"roll_{pass_num}.jsonl").read_text().splitlines()
        if s.strip()
    ]
    by_model = {}
    for row in rows:
        key = (row["id"], row["seed"])
        by_model.setdefault(row["model"], {})[key] = row
    assert set(by_model) == {"original", "edited"}
    assert by_model["original"].keys() == by_model["edited"].keys()
    pairs = [
        (by_model["original"][key], by_model["edited"][key])
        for key in sorted(by_model["original"])
    ]
    batch_idx = pass_num - 36
    assert {a["id"] for a, b in pairs} == set(batches[batch_idx])
    return pairs

def paired_summary(pairs):
    original = np.array([int(a["success"]) for a, b in pairs])
    changed = np.array([int(b["success"]) for a, b in pairs])
    delta = changed - original
    return {
        "n": len(delta),
        "original": int(original.sum()),
        "edited": int(changed.sum()),
        "wins": int((delta == 1).sum()),
        "losses": int((delta == -1).sum()),
        "delta_pp": float(100 * delta.mean()),
        "paired_se_pp": float(100 * delta.std(ddof=1) / np.sqrt(len(delta))),
    }

batch4_summary = paired_summary(read_pairs(39))
cumulative = paired_summary([p for k in range(36, 40) for p in read_pairs(k)])
print("Batch 4:", batch4_summary)
print("Cumulative:", cumulative)

with (work / "notes.md").open("a") as f:
    f.write(
        "\n## Final confirmation batch4 (pass39)\n"
        "73/128->77/128 (+3.1 +/-2.2pp paired SE),6wins/2losses. "
        "Environment+2,position+1,semantic+1,object/task unchanged. "
        "Cumulative306/512->309/512; still inconclusive. "
        "Candidate and final batch remain frozen.\n"
    )

print("FROZEN BATCH 5 REQUEST:")
print(json.dumps(requests[4]))
```
