Across both Sensor Noise batches, the edit lost seven net successes out of 256 pairs, about −2.7 percentage points. I will record this additional safety concern and load the frozen Camera Viewpoints batch.
```python
def episode_keys(episodes):
    return {(e["suite"], e["id"], e["seed"]) for e in episodes}

def verify_frozen_candidate():
    assert edited() == [name]
    h = hashlib.sha256(load(name).contiguous().numpy().tobytes()).hexdigest()
    assert h == ledger["tensor_sha256"]

def log_safety_pass(k):
    global ledger
    item = safety_plan[k - 27]
    rows = read_roll(k)
    expected = episode_keys(item["episodes"])
    for model in ("original", "edited"):
        actual = episode_keys([r for r in rows if r["model"] == model])
        assert actual == expected, (k, model, "Plan mismatch")
    assert {r["category"] for r in rows} == {item["category"]}
    verify_frozen_candidate()
    category_passes = [
        27 + i for i, x in enumerate(safety_plan)
        if x["category"] == item["category"] and 27 + i <= k
    ]
    one, combined = paired_summary([k]), paired_summary(category_passes)
    ledger = json.loads(ledger_path.read_text())
    ledger["safety"][item["category"]] = combined
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    with (root / "notes.md").open("a") as f:
        f.write(f"\n## Pass {k}: safety {item['category']}, batch {item['batch']}\n"
                + "- Batch: " + json.dumps(one) + "\n"
                + "- Category cumulative: " + json.dumps(combined) + "\n"
                + "- Candidate unchanged; no outcome-dependent tuning.\n")
    print("Batch:", json.dumps(one))
    print(item["category"] + " cumulative:", json.dumps(combined))

def emit_request(k, episodes):
    request = {"type": "rollout", "model": "both", "episodes": episodes}
    assert len(episodes) <= 128 and len(episode_keys(episodes)) == len(episodes)
    (root / f"request_{k}.json").write_text(json.dumps(request))
    print("NEXT REQUEST:")
    print(json.dumps(request, separators=(",", ":")))

log_safety_pass(32)
assert safety_plan[6]["category"] == "Camera Viewpoints"
emit_request(33, safety_plan[6]["episodes"])
```
