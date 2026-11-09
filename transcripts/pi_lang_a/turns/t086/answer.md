The first replication batch gained 4 successes out of 128, with 5 rescues and 1 regression. I will verify that it matched the saved request, record the result, and load the next batch without changing the candidate.
```python
def audit_request(k):
    req = json.loads((W / f"request{k}.json").read_text())
    expected = {(e["suite"], int(e["id"]), int(e["seed"])) for e in req["episodes"]}
    actual = set(pair_records(read_roll(k)))
    assert actual == expected, {"pass": k, "missing": sorted(expected-actual),
                                "unexpected": sorted(actual-expected)}
    assert all(set(v) == {"original", "edited"} for v in pair_records(read_roll(k)).values())
    return {"pass": k, "n": len(actual), "manifest_exact": True}

def log_replication_and_next(k, next_k):
    audit = audit_request(k)
    stats = paired_stats(read_roll(k))
    print("Completed:", json.dumps({**audit, **stats}, separators=(",", ":")))
    notes = (W / "notes.md").read_text()
    import re
    notes = re.sub(r"Forward passes used: \d+/50\.", f"Forward passes used: {k}/50.", notes)
    notes += f"\nPass{k} frozen18 second-seed replication: {json.dumps(stats)}. Manifest exact; no weight changes.\n"
    (W / "notes.md").write_text(notes)
    print(f"REQUEST{next_k}")
    print((W / f"request{next_k}.json").read_text())

log_replication_and_next(41, 42)
```
