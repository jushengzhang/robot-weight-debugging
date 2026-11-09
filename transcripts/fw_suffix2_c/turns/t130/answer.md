The first clean batch lost three successes and gained none. That is a warning, not yet a precise estimate of clean-performance damage. I’ll log it and prepare the second matched clean batch without changing the candidate.
```python
def record_validation(pass_id, label, pooled_passes, result_key):
    audit_rollout(pass_id)
    single = paired_stats([pass_id])
    pooled = paired_stats(pooled_passes)
    print(label, json.dumps(single))
    print("POOLED_" + result_key.upper(), json.dumps(pooled))
    results = json.loads((root / "validation_results.json").read_text())
    results[result_key] = pooled
    results.setdefault("by_pass", {})[str(pass_id)] = single
    (root / "validation_results.json").write_text(json.dumps(results, indent=2))
    with (root / "notes.md").open("a") as f:
        f.write(f"\nFrozen validation {label}/pass{pass_id}: " +
                json.dumps(single) + ". Pooled: " + json.dumps(pooled) +
                ". No weight changes.\n")
    return pooled

def show_next_request(pass_id):
    request = json.loads((root / f"request{pass_id}.json").read_text())
    assert request["model"] == "both" and len(request["episodes"]) == 128
    if 44 <= pass_id <= 47:
        raw_request = json.loads(
            (root / f"request{pass_id - 4}.json").read_text())
        assert request["episodes"] == raw_request["episodes"]
        assert request["language"] == "strip"
    print(f"REQUEST{pass_id}")
    print(json.dumps(request, separators=(",", ":")))

record_validation(44, "strip_A", [44], "clean")
show_next_request(45)
```
