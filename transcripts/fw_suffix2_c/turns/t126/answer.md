Batch C gained one success, but the first three batches together still show no improvement. I’ll record the paired estimates and load the final preselected tagged batch; the weights will remain unchanged.
```python
exec(helper_text, globals())

def audit_rollout(pass_id):
    request = json.loads((root / f"request{pass_id}.json").read_text())
    expected = {(r["suite"], r["id"], r["seed"]) for r in request["episodes"]}
    assert len(expected) == len(request["episodes"]) == 128
    rows = [json.loads(line) for line in
            (root / f"roll_{pass_id}.jsonl").read_text().splitlines()]
    for model in ("original", "edited"):
        observed = [(r["suite"], r["id"], r["seed"])
                    for r in rows if r["model"] == model]
        assert len(observed) == 128 and set(observed) == expected, (pass_id, model)
    return rows

for k in (40, 41, 42):
    audit_rollout(k)
s42 = paired_stats([42])
pooled_raw = paired_stats([40, 41, 42])
print("C", json.dumps(s42))
print("POOLED_RAW", json.dumps(pooled_raw))
with (root / "notes.md").open("a") as f:
    f.write("\nFrozen validation raw C/pass42: " + json.dumps(s42) +
            ". Pooled A-C: " + json.dumps(pooled_raw) +
            ". Frozen tensor hashes verified after restart; no weight changes.\n")
print("REQUEST43")
print(json.dumps(json.loads((root / "request43.json").read_text()),
                 separators=(",", ":")))
```
